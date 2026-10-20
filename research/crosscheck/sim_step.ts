/**
 * Cross-check: drive a wheel-speed step profile in a third-party simulator
 * and write the trace as CSV.
 *
 * This file is copied into the simulator's checkout and run there, because it
 * imports that project's own modules. It is measurement code: it changes
 * nothing in OVERBOARD and nothing in the simulator.
 *
 * The profile is the one a planner cares about: run up to a steady speed,
 * then stop. The stopping distance is a property of the motor and the tyres,
 * not of the controller, so it is the thing worth checking against another
 * model.
 *
 * Usage (inside the simulator's checkout):
 *   npx tsx crosscheck.ts <out.csv> <command_dps> <stop_mode> <run_ms>
 *
 *   command_dps  the commanded wheel speed, in wheel degrees per second
 *   stop_mode    "brake" or "coast"
 *   run_ms       how long the drive phase lasts
 */

import { writeFileSync } from "node:fs";
import season from "./seasons/2026-27/season.json";
import {
  PORTS,
  Simulation,
  SpikeApi,
  Stop,
  makeDriveBase,
} from "./packages/sim/src/index";

const outPath = process.argv[2] ?? "crosscheck_trace.csv";
const commandDps = Number(process.argv[3] ?? 1110);
const stopMode = (process.argv[4] ?? "brake").toLowerCase();
const runMs = Number(process.argv[5] ?? 1500);
const totalMs = runMs + 1500;
const dtMs = 10;

async function main() {
  const sim = await Simulation.create({
    season: season as never,
    robot: makeDriveBase(),
    // Well clear of the walls and of every mission model.
    start: { xMm: 500, yMm: 300, headingDeg: 0 },
    footprints: false,
  });
  // Let the robot settle on the mat before it is measured.
  sim.stepMs(500);

  const api = new SpikeApi(sim);
  api.pair(0, 0, 1);
  const left = sim.motor(PORTS[0]);
  const right = sim.motor(PORTS[1]);
  if (!left || !right) throw new Error("the drive base has no motors");

  const rows: string[] = [
    "t_ms,cmd_dps,left_dps,right_dps,left_duty,x_mm,y_mm,heading_deg,stop_mode",
  ];
  const startY = sim.robotPose().yMm;

  for (let t = 0; t <= totalMs; t += dtMs) {
    if (t === 0) {
      // A big acceleration so the profile is not the thing being measured.
      api.pairMove(0, 0, commandDps, 10000);
    }
    if (t === runMs) {
      api.pairStop(0, stopMode === "coast" ? Stop.COAST : Stop.BRAKE);
    }
    sim.stepMs(dtMs);
    const pose = sim.robotPose();
    rows.push(
      [
        t,
        t < runMs ? commandDps : 0,
        left.speed.toFixed(4),
        right.speed.toFixed(4),
        left.duty.toFixed(4),
        pose.xMm.toFixed(4),
        pose.yMm.toFixed(4),
        pose.headingDeg.toFixed(4),
        stopMode,
      ].join(","),
    );
  }

  writeFileSync(outPath, rows.join("\n") + "\n");
  const endY = sim.robotPose().yMm;
  console.log(
    `wrote ${outPath}: ${rows.length - 1} rows, ` +
      `start y ${startY.toFixed(1)} mm, end y ${endY.toFixed(1)} mm`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
