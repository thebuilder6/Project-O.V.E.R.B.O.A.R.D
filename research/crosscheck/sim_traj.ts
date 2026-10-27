/**
 * Cross-check: drive a closed-loop trajectory in a third-party simulator
 * and write the pose trace as CSV (Q25, B1).
 *
 * This file is copied into the simulator's checkout and run there (see
 * run_traj.sh), because it imports that project's own modules. It is
 * measurement code: it changes nothing in OVERBOARD and nothing in the
 * simulator.
 *
 * The loop is the deployed one (pybricks_code/ramsete.py, ported line for
 * line): reference interpolation with the 10 ms tick-midpoint lead, Ramsete
 * feedback on the robot's own integrated odometry (encoders only, never the
 * true pose), the KinematicLimiter, then per-wheel speed commands held over
 * the 20 ms heartbeat (fll-sim steps at 10 ms).
 *
 * Usage (inside the simulator's checkout):
 *   npx tsx crosscheck_traj.ts <spec.json | --straight-turn> <out.csv> [mode]
 *
 *   mode  "closedloop" (default) runs the deployed loop;
 *         "openloop" commands the reference (vl, vr) directly, no feedback.
 *
 * Spec JSON: { wheelDiameterMm, trackWidthMm, robot: {...limiter config...},
 *              start: { xMm, yMm, headingDeg },
 *              samples: [{ t, x, y, heading, vl, vr, omega }] }  (SI units)
 *
 * Trace CSV (SI units; the step harness trace.csv stays in mm):
 *   t_ms,tick,ref_x_m,ref_y_m,ref_heading_rad,odom_x_m,odom_y_m,
 *   odom_heading_rad,true_x_m,true_y_m,true_heading_rad,
 *   cmd_vl_dps,cmd_vr_dps,left_dps,right_dps,left_deg,right_deg
 *
 * The spec's samples are in the planner frame (heading 0 = +x, CCW positive).
 * The simulator's mat frame has heading 0 = +y (world.ts robotPose), so the
 * driver places the robot with mat heading = planner heading - 90 deg (same
 * positions) and reads poses back with planner heading = mat heading + 90 deg;
 * the trace is entirely in the planner frame.
 */

import { readFileSync, writeFileSync } from "node:fs";
import season from "./seasons/2026-27/season.json";
import {
  PORTS,
  Simulation,
  SpikeApi,
  Stop,
  makeDriveBase,
} from "./packages/sim/src/index";

interface Sample {
  t: number;
  x: number;
  y: number;
  heading: number;
  vl: number;
  vr: number;
  omega: number;
}

interface Spec {
  wheelDiameterMm: number;
  trackWidthMm: number;
  robot?: Record<string, number>;
  samples: Sample[];
}

/** Straight 0.5 m at 0.3 m/s, hold, in-place +90 deg turn, hold (sanity gate 4). */
function straightTurnSpec(): Spec {
  const dt = 0.02;
  const trackM = 0.12;
  // Smooth plan-like profile (0.2 s ramps at 1.5 m/s^2) on a uniform grid:
  // a jumped speed reference with fll-sim's fast profile generator
  // destabilises its motor servo (measured in the spike).
  const vAt = (t: number) => {
    if (t < 0.2) return 0;
    if (t < 0.4) return ((t - 0.2) / 0.2) * 0.3;
    if (t < 1.85) return 0.3;
    if (t < 2.05) return (0.3 * (2.05 - t)) / 0.2;
    return 0;
  };
  const wAt = (t: number) => (t >= 2.25 && t < 3.29 ? Math.PI / 2 / 1.04 : 0);
  const samples: Sample[] = [];
  let x = 0.5, y = 0.3, h = 0;
  const total = 3.79;
  for (let t = 0; t <= total + 1e-9; t += dt) {
    const tt = Math.round(t * 1000) / 1000;
    const v = vAt(tt), omega = wAt(tt);
    samples.push({
      t: tt,
      x, y, heading: h,
      vl: v - (omega * trackM) / 2,
      vr: v + (omega * trackM) / 2,
      omega,
    });
    x += v * Math.cos(h) * dt;
    y += v * Math.sin(h) * dt;
    h += omega * dt;
  }
  return {
    wheelDiameterMm: 56,
    trackWidthMm: 120,
    // fll-sim's own drive base numbers, for the limiter (RQ-B).
    robot: {
      mass: 0.598, inertia: 0.0012, track_width: 0.12, wheel_radius: 0.028,
      v_max_rad_s: 22.51, t_max_nm: 0.18, gearing: 1.0, cof: 1.0, h_cg: 0.05,
    },
    samples,
  };
}

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/** RamseteController (pybricks_code/ramsete.py), ported. */
class RamseteController {
  constructor(readonly b = 2.0, readonly zeta = 0.7) {}
  calculate(
    x: number, y: number, theta: number,
    xr: number, yr: number, thetar: number,
    refV: number, refOmega: number,
  ): [number, number] {
    const exG = xr - x, eyG = yr - y;
    const ex = Math.cos(theta) * exG + Math.sin(theta) * eyG;
    const ey = -Math.sin(theta) * exG + Math.cos(theta) * eyG;
    const etheta = normalizeAngle(thetar - theta);
    const k1 = 2 * this.zeta * Math.sqrt(refOmega * refOmega + this.b * refV * refV);
    const vCmd = refV * Math.cos(etheta) + k1 * ex;
    const sinc = Math.abs(etheta) < 1e-6 ? 1.0 : Math.sin(etheta) / etheta;
    const omegaCmd = refOmega + this.b * refV * sinc * ey + k1 * etheta;
    return [vCmd, omegaCmd];
  }
}

/** KinematicLimiter (pybricks_code/ramsete.py), ported; no backlash (KI-025/KI-028). */
class KinematicLimiter {
  private cfg: Record<string, number>;
  constructor(cfg: Record<string, number>) { this.cfg = cfg; }
  private get(k: string, d: number) { return this.cfg[k] ?? d; }

  maxForce(vWheel: number): number {
    const omega = (Math.abs(vWheel) / this.get("wheel_radius", 0.028)) * this.get("gearing", 1.0);
    const torque = this.get("t_max_nm", 0.138) * Math.max(0, 1 - omega / this.get("v_max_rad_s", 19.4));
    return (torque / this.get("wheel_radius", 0.028)) * this.get("gearing", 1.0);
  }

  limit(vCmd: number, omegaCmd: number, currentV: number, currentOmega: number, dt: number): [number, number] {
    if (dt <= 0) return [vCmd, omegaCmd];
    const mass = this.get("mass", 1.5), inertia = this.get("inertia", 0.001);
    const L = this.get("track_width", 0.0965), cof = this.get("cof", 0.6);
    const g = 9.81, hCg = this.get("h_cg", 0.05);

    const vlCmd = vCmd - (omegaCmd * L) / 2;
    const vrCmd = vCmd + (omegaCmd * L) / 2;
    const vlCurr = currentV - (currentOmega * L) / 2;
    const vrCurr = currentV + (currentOmega * L) / 2;

    const alReq = (vlCmd - vlCurr) / dt;
    const arReq = (vrCmd - vrCurr) / dt;
    const aReq = (alReq + arReq) / 2;
    const alphaReq = (arReq - alReq) / L;

    const fTotal = mass * aReq;
    const mTotal = inertia * alphaReq;
    let frReq = (fTotal + (2 * mTotal) / L) / 2;
    let flReq = fTotal - frReq;

    const fMaxL = this.maxForce(vlCurr), fMaxR = this.maxForce(vrCurr);
    const scaleF = Math.min(1,
      Math.abs(flReq) > fMaxL ? fMaxL / Math.abs(flReq) : 1,
      Math.abs(frReq) > fMaxR ? fMaxR / Math.abs(frReq) : 1);
    flReq *= scaleF; frReq *= scaleF;

    // Per-wheel traction limit with lateral weight transfer (KI-010).
    const centripetal = currentV * currentOmega;
    const baseN = (mass * g) / 2;
    const transfer = (centripetal * (mass * hCg)) / L;
    const nl = Math.max(0, baseN - transfer), nr = Math.max(0, baseN + transfer);
    const trMaxL = cof * nl, trMaxR = cof * nr;
    const scaleTr = Math.min(1,
      Math.abs(flReq) > trMaxL ? trMaxL / Math.abs(flReq) : 1,
      Math.abs(frReq) > trMaxR ? trMaxR / Math.abs(frReq) : 1);
    flReq *= scaleTr; frReq *= scaleTr;

    const fAch = flReq + frReq;
    const mAch = (frReq - flReq) * (L / 2);
    const aAch = fAch / mass;
    const alphaAch = mAch / inertia;
    const alAch = aAch - (alphaAch * L) / 2;
    const arAch = aAch + (alphaAch * L) / 2;
    const vlAch = vlCurr + alAch * dt;
    const vrAch = vrCurr + arAch * dt;
    return [(vlAch + vrAch) / 2, (vrAch - vlAch) / L];
  }
}

/** Odometry from the encoders only (the robot's own DriveBase.state() estimate). */
class Odometry {
  x: number; y: number; heading: number;
  private lastDist: number; private lastYaw: number;

  constructor(private readEnc: () => { distM: number; yawRad: number }, x0: number, y0: number, h0: number) {
    this.x = x0; this.y = y0; this.heading = h0;
    const e = readEnc();
    this.lastDist = e.distM; this.lastYaw = e.yawRad;
  }

  update() {
    const e = this.readEnc();
    const dDist = e.distM - this.lastDist;
    const dAngle = normalizeAngle(e.yawRad - this.lastYaw);
    const avg = this.heading + dAngle / 2;
    this.x += dDist * Math.cos(avg);
    this.y += dDist * Math.sin(avg);
    this.heading = normalizeAngle(this.heading + dAngle);
    this.lastDist = e.distM; this.lastYaw = e.yawRad;
  }
}

async function main() {
  const specArg = process.argv[2] ?? "--straight-turn";
  const outPath = process.argv[3] ?? "traj_trace.csv";
  const mode = (process.argv[4] ?? "closedloop").toLowerCase();

  const spec: Spec = specArg === "--straight-turn"
    ? straightTurnSpec()
    : JSON.parse(readFileSync(specArg, "utf8"));

  const sim = await Simulation.create({
    season: season as never,
    robot: makeDriveBase({
      wheelDiameterMm: spec.wheelDiameterMm,
      trackWidthMm: spec.trackWidthMm,
    }),
    // The spec's samples are in the planner frame (heading 0 = +x, CCW+);
    // the sim's mat frame has heading 0 = +y (world.ts robotPose). The map is
    // identity positions with mat heading = planner heading - 90 deg, so the
    // trace reads back in the planner frame and the twin prediction needs no
    // conversion.
    start: {
      xMm: spec.samples[0].x * 1000,
      yMm: spec.samples[0].y * 1000,
      headingDeg: (spec.samples[0].heading * 180) / Math.PI - 90,
    },
    footprints: false,
  });
  // Let the robot settle on the mat before it is measured.
  sim.stepMs(500);

  const api = new SpikeApi(sim);
  api.pair(0, 0, 1);
  const left = sim.motor(PORTS[0]);
  const right = sim.motor(PORTS[1]);
  if (!left || !right) throw new Error("the drive base has no motors");

  const wheelDiaM = spec.wheelDiameterMm / 1000;
  const trackM = spec.trackWidthMm / 1000;
  const degPerM = 360 / (Math.PI * wheelDiaM);
  // The left motor is mirrored in the pair (positive = forward): its encoder
  // runs backwards when the robot drives forwards (pinned by the spike run).
  const readEnc = () => {
    const leftFwdDeg = -left.relativePosition();
    const rightFwdDeg = right.relativePosition();
    const leftFwdM = (leftFwdDeg / 360) * Math.PI * wheelDiaM;
    const rightFwdM = (rightFwdDeg / 360) * Math.PI * wheelDiaM;
    return {
      distM: (leftFwdM + rightFwdM) / 2,
      yawRad: (rightFwdM - leftFwdM) / trackM,
    };
  };

  const ctrl = new RamseteController(2.0, 0.7);
  const limiter = new KinematicLimiter(spec.robot ?? {});
  const samples = spec.samples;
  const s0 = samples[0];
  const odom = new Odometry(readEnc, s0.x, s0.y, s0.heading);

  const rows: string[] = [
    "t_ms,tick,ref_x_m,ref_y_m,ref_heading_rad,odom_x_m,odom_y_m,odom_heading_rad," +
    "true_x_m,true_y_m,true_heading_rad,cmd_vl_dps,cmd_vr_dps,left_dps,right_dps,left_deg,right_deg",
  ];

  const dtMs = 10;
  const tickMs = 20;
  const totalT = samples[samples.length - 1].t;
  let sampleIdx = 0;
  let currentV = 0, currentOmega = 0;
  let lastT = 0;
  let cmdL = 0, cmdR = 0;
  let refX = s0.x, refY = s0.y, refH = s0.heading;

  const runMs = Math.round((totalT + 0.3) * 1000);
  for (let tMs = 0; tMs <= runMs; tMs += dtMs) {
    const t = tMs / 1000;
    const onTick = tMs % tickMs === 0;

    if (t <= totalT) {
      if (onTick) {
        while (sampleIdx < samples.length - 2 && samples[sampleIdx + 1].t <= t) sampleIdx++;
        const s1 = samples[sampleIdx], s2 = samples[sampleIdx + 1];
        const alpha = s2.t === s1.t ? 0 : (t - s1.t) / (s2.t - s1.t);
        refX = s1.x + alpha * (s2.x - s1.x);
        refY = s1.y + alpha * (s2.y - s1.y);
        const dh = normalizeAngle(s2.heading - s1.heading);
        refH = normalizeAngle(s1.heading + alpha * dh);

        // Reference velocities with the tick midpoint lead (T/2 = 10 ms).
        const v1 = (s1.vl + s1.vr) / 2, v2 = (s2.vl + s2.vr) / 2;
        const dtSeg = s2.t - s1.t;
        const tMid = t + 0.010;
        const isFinalStop = sampleIdx >= samples.length - 2 && Math.abs(v2) < 1e-4;
        let refV = 0, refOmega = 0;
        if (isFinalStop && tMid >= s2.t) {
          refV = 0; refOmega = 0;
        } else if (dtSeg > 0) {
          const aMid = Math.min(Math.max((tMid - s1.t) / dtSeg, 0), 1);
          refV = v1 + aMid * (v2 - v1);
          refOmega = dh / dtSeg;
        }

        odom.update();
        let vCmd: number, omegaCmd: number;
        if (mode === "openloop") {
          vCmd = refV; omegaCmd = refOmega;
        } else {
          [vCmd, omegaCmd] = ctrl.calculate(odom.x, odom.y, odom.heading, refX, refY, refH, refV, refOmega);
          const dt = t - lastT;
          if (dt > 0) [vCmd, omegaCmd] = limiter.limit(vCmd, omegaCmd, currentV, currentOmega, dt);
        }
        currentV = vCmd; currentOmega = omegaCmd;
        lastT = t;
        const vl = vCmd - (omegaCmd * trackM) / 2;
        const vr = vCmd + (omegaCmd * trackM) / 2;
        cmdL = vl * degPerM;
        cmdR = vr * degPerM;
        // Retarget once per tick and let the pair hold between (the deployed
        // drive() boundary is the 20 ms heartbeat, not every sim step).
        api.pairMoveTank(0, cmdL, cmdR, 10000);
      }
    } else if (onTick) {
      // The deployed loop has ended: stop() coasts (brake is never a reference).
      api.pairStop(0, Stop.COAST);
    }

    sim.stepMs(dtMs);
    const pose = sim.robotPose();
    rows.push(
      [
        tMs,
        onTick ? 1 : 0,
        refX.toFixed(6), refY.toFixed(6), refH.toFixed(6),
        odom.x.toFixed(6), odom.y.toFixed(6), odom.heading.toFixed(6),
        (pose.xMm / 1000).toFixed(6), (pose.yMm / 1000).toFixed(6),
        normalizeAngle(((pose.headingDeg + 90) * Math.PI) / 180).toFixed(6),
        cmdL.toFixed(3), cmdR.toFixed(3),
        left.speed.toFixed(4), right.speed.toFixed(4),
        left.relativePosition(), right.relativePosition(),
      ].join(","),
    );
  }

  writeFileSync(outPath, rows.join("\n") + "\n");
  const pose = sim.robotPose();
  const last = samples[samples.length - 1];
  const trueH = normalizeAngle(((pose.headingDeg + 90) * Math.PI) / 180);
  console.log(
    `wrote ${outPath}: ${rows.length - 1} rows, mode ${mode}\n` +
    `expected end pose: x ${last.x.toFixed(4)} y ${last.y.toFixed(4)} h ${last.heading.toFixed(4)}\n` +
    `true end pose:     x ${(pose.xMm / 1000).toFixed(4)} y ${(pose.yMm / 1000).toFixed(4)} h ${trueH.toFixed(4)}\n` +
    `odom end pose:     x ${odom.x.toFixed(4)} y ${odom.y.toFixed(4)} h ${odom.heading.toFixed(4)}`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
