"""
Convergence Visualization for Trajectory Optimization

Displays real-time trajectory convergence with three display modes:
- Parallel mode: Show all TEB/STOMP heuristics simultaneously
- Best mode: Show only best trajectory improving through phases
- Layered mode: Show initial guess, then overlay each phase result as layers

Uses Plotly for interactive web-based visualization.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


class ConvergencePlotter:
    """Visualizes trajectory optimization convergence over iterations using Plotly."""
    
    def __init__(self, mode='best'):
        """
        Initialize convergence plotter.
        
        Args:
            mode: Visualization mode ('parallel', 'best', or 'layered')
        """
        self.mode = mode
        self.fig = None
        self.iteration_history = []
        
    def plot_convergence(self, iteration_history, waypoints=None, title="Trajectory Convergence", output_file=None):
        """
        Plot convergence history using Plotly.
        
        Args:
            iteration_history: List of iteration dictionaries with 'cost', 'trajectory', 'phase'
            waypoints: Optional list of waypoint tuples for reference
            title: Plot title
            output_file: Optional path to save HTML file. If None, opens in browser.
        """
        self.iteration_history = iteration_history
        
        # Create figure with two subplots (trajectory and cost)
        self.fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Trajectory Evolution', 'Cost Function Value'),
            vertical_spacing=0.05
        )
        
        self.fig.update_layout(
            title=f"{title} - {self.mode.capitalize()} Mode",
            height=800,
            showlegend=True
        )
        
        # Plot waypoints if provided
        if waypoints is not None:
            wx = [wp[0] for wp in waypoints]
            wy = [wp[1] for wp in waypoints]
            self.fig.add_trace(
                go.Scatter(x=wx, y=wy, mode='lines+markers', 
                          name='Waypoints', marker=dict(size=10, color='red'),
                          line=dict(width=2, color='red')),
                row=1, col=1
            )
        
        # Plot based on mode
        if self.mode == 'parallel':
            self._plot_parallel()
        elif self.mode == 'best':
            self._plot_best()
        elif self.mode == 'layered':
            self._plot_layered()
        
        # Plot cost evolution
        self._plot_cost()
        
        # Update axis labels
        self.fig.update_xaxes(title_text="X (m)", row=1, col=1)
        self.fig.update_yaxes(title_text="Y (m)", row=1, col=1)
        self.fig.update_xaxes(title_text="Iteration", row=2, col=1)
        self.fig.update_yaxes(title_text="Cost (s)", row=2, col=1)
        
        # Output
        if output_file:
            self.fig.write_html(output_file)
        else:
            self.fig.show()
    
    def _plot_parallel(self):
        """Plot all heuristics simultaneously with different colors."""
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']
        color_idx = 0
        
        for i, entry in enumerate(self.iteration_history):
            traj = entry['trajectory']
            phase = entry.get('phase', 'unknown')
            heuristic_type = entry.get('heuristic_type', None)
            
            # Choose color based on heuristic type or phase
            if heuristic_type == 'TEB':
                color = 'blue'
            elif heuristic_type == 'STOMP':
                color = 'green'
            elif phase == 'bootstrap':
                color = 'gray'
            elif phase == 'global_solve':
                color = 'orange'
            elif phase == 'final_polish':
                color = 'red'
            else:
                color = colors[color_idx % len(colors)]
                color_idx += 1
            
            # Plot trajectory
            x = traj[:, 0]
            y = traj[:, 1]
            label = f"{phase}" if heuristic_type is None else f"{heuristic_type}"
            self.fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', name=label,
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
    
    def _plot_best(self):
        """Plot only best trajectory improving through phases."""
        # Find best trajectory at each phase
        phases = {}
        for entry in self.iteration_history:
            phase = entry.get('phase', 'unknown')
            cost = entry['cost']
            
            if phase not in phases or cost < phases[phase]['cost']:
                phases[phase] = entry
        
        # Plot in order of phases
        phase_order = ['bootstrap', 'global_solve', 'final_polish']
        colors = {'bootstrap': 'gray', 'global_solve': 'orange', 'final_polish': 'red'}
        
        for phase in phase_order:
            if phase in phases:
                entry = phases[phase]
                traj = entry['trajectory']
                x = traj[:, 0]
                y = traj[:, 1]
                iteration = entry['iteration']
                cost = entry['cost']
                
                self.fig.add_trace(
                    go.Scatter(x=x, y=y, mode='lines+markers',
                              name=f"{phase} (iter={iteration}, cost={cost:.3f}s)",
                              line=dict(color=colors.get(phase, 'blue'), width=2),
                              marker=dict(size=4)),
                    row=1, col=1
                )
    
    def _plot_layered(self):
        """Plot initial guess, then overlay each phase result as layers."""
        # Sort by iteration
        sorted_entries = sorted(self.iteration_history, key=lambda x: x['iteration'])
        
        # Plot as layers with increasing opacity
        for i, entry in enumerate(sorted_entries):
            traj = entry['trajectory']
            phase = entry.get('phase', 'unknown')
            x = traj[:, 0]
            y = traj[:, 1]
            
            # Increasing opacity for later iterations
            opacity = 0.3 + 0.7 * (i / len(sorted_entries))
            linewidth = 1 + 2 * (i / len(sorted_entries))
            
            # Use a color scale
            colorscale_value = i / len(sorted_entries)
            color = f'rgb({int(255 * (1 - colorscale_value))}, {int(255 * colorscale_value)}, 128)'
            
            self.fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines',
                          name=f"{phase} (iter={entry['iteration']})",
                          line=dict(color=color, width=linewidth),
                          opacity=opacity),
                row=1, col=1
            )
    
    def _plot_cost(self):
        """Plot cost function value over iterations."""
        iterations = []
        costs = []
        phases = []
        
        for entry in self.iteration_history:
            iterations.append(entry['iteration'])
            costs.append(entry['cost'])
            phases.append(entry.get('phase', 'unknown'))
        
        # Plot cost curve
        self.fig.add_trace(
            go.Scatter(x=iterations, y=costs, mode='lines+markers',
                      name='Cost', line=dict(color='blue', width=2),
                      marker=dict(size=6)),
            row=2, col=1
        )
        
        # Annotate phase transitions
        phase_colors = {'bootstrap': 'gray', 'global_solve': 'orange', 
                       'final_polish': 'red', 'initial_guess': 'lightblue',
                       'refinement': 'green'}
        
        for i, phase in enumerate(phases):
            if phase in phase_colors:
                self.fig.add_trace(
                    go.Scatter(x=[iterations[i]], y=[costs[i]], mode='markers',
                              name=f'{phase} marker', marker=dict(color=phase_colors[phase], size=10),
                              showlegend=False),
                    row=2, col=1
                )
    
    def animate_convergence(self, iteration_history, waypoints=None, title="Trajectory Convergence", 
                          interval=500, save_path=None):
        """
        Animate convergence over iterations using Plotly.
        
        Args:
            iteration_history: List of iteration dictionaries
            waypoints: Optional list of waypoint tuples
            title: Animation title
            interval: Animation interval in milliseconds (not used in Plotly)
            save_path: Optional path to save animation as HTML
        """
        self.iteration_history = iteration_history
        
        # Create figure
        self.fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Trajectory Evolution', 'Cost Function Value'),
            vertical_spacing=0.05
        )
        
        self.fig.update_layout(
            title=f"{title} - Animation",
            height=800,
            showlegend=True
        )
        
        # Plot waypoints
        if waypoints is not None:
            wx = [wp[0] for wp in waypoints]
            wy = [wp[1] for wp in waypoints]
            self.fig.add_trace(
                go.Scatter(x=wx, y=wy, mode='lines+markers', 
                          name='Waypoints', marker=dict(size=10, color='red'),
                          line=dict(width=2, color='red')),
                row=1, col=1
            )
        
        # Create frames for animation
        frames = []
        for i, entry in enumerate(iteration_history):
            traj = entry['trajectory']
            cost = entry['cost']
            phase = entry.get('phase', 'unknown')
            iteration_num = entry['iteration']
            
            # Trajectory frame
            x = traj[:, 0]
            y = traj[:, 1]
            
            # Cost data up to this frame
            costs_so_far = [iteration_history[j]['cost'] for j in range(i + 1)]
            iterations_so_far = [iteration_history[j]['iteration'] for j in range(i + 1)]
            
            frame = go.Frame(
                data=[
                    go.Scatter(x=x, y=y, mode='lines', name='Trajectory',
                              line=dict(color='blue', width=2)),
                    go.Scatter(x=iterations_so_far, y=costs_so_far, mode='lines+markers',
                              name='Cost', line=dict(color='blue', width=2),
                              marker=dict(size=6))
                ],
                name=f"frame_{i}"
            )
            frames.append(frame)
        
        # Add initial traces
        if iteration_history:
            first_traj = iteration_history[0]['trajectory']
            self.fig.add_trace(
                go.Scatter(x=first_traj[:, 0], y=first_traj[:, 1], mode='lines',
                          name='Trajectory', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            self.fig.add_trace(
                go.Scatter(x=[iteration_history[0]['iteration']], y=[iteration_history[0]['cost']],
                          mode='markers', name='Cost', marker=dict(size=6)),
                row=2, col=1
            )
        
        # Add animation
        self.fig.update(frames=frames)
        
        # Update axis labels
        self.fig.update_xaxes(title_text="X (m)", row=1, col=1)
        self.fig.update_yaxes(title_text="Y (m)", row=1, col=1)
        self.fig.update_xaxes(title_text="Iteration", row=2, col=1)
        self.fig.update_yaxes(title_text="Cost (s)", row=2, col=1)
        
        # Add play button
        self.fig.update_layout(
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate",
                                                           args=[None, {"frame": {"duration": 500, "redraw": True},
                                                                     "fromcurrent": True}])])]
        )
        
        # Output
        if save_path:
            self.fig.write_html(save_path)
        else:
            self.fig.show()
        
        return self.fig


def plot_convergence(iteration_history, mode='best', waypoints=None, title="Trajectory Convergence", output_file=None):
    """
    Convenience function to plot convergence.
    
    Args:
        iteration_history: List of iteration dictionaries
        mode: Visualization mode ('parallel', 'best', or 'layered')
        waypoints: Optional list of waypoint tuples
        title: Plot title
        output_file: Optional path to save HTML file. If None, opens in browser.
    """
    plotter = ConvergencePlotter(mode=mode)
    plotter.plot_convergence(iteration_history, waypoints=waypoints, title=title, output_file=output_file)


def animate_convergence(iteration_history, waypoints=None, title="Trajectory Convergence",
                       save_path=None):
    """
    Convenience function to animate convergence.
    
    Args:
        iteration_history: List of iteration dictionaries
        waypoints: Optional list of waypoint tuples
        title: Animation title
        save_path: Optional path to save animation as HTML
    """
    plotter = ConvergencePlotter()
    return plotter.animate_convergence(iteration_history, waypoints=waypoints, 
                                      title=title, save_path=save_path)
