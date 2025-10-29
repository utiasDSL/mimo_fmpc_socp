#!/usr/bin/env python3
"""
Monte Carlo Data Viewer GUI

A GUI application for visualizing Monte Carlo experiment results.
Allows browsing experiment runs and plotting states/inputs for individual trials.

Prerequisites:
    - Must be run in the 'safe' conda environment
    - Requires: numpy, matplotlib, tkinter (usually included with Python)

Usage:
    conda activate safe
    python monte_carlo_viewer.py

Features:
    - Browse and load Monte Carlo experiment runs
    - Select controller (NMPC, FMPC, FMPC+SOCP)
    - Select individual trials
    - Plot states or inputs for each trial
    - View trial information including initial conditions
"""

import os
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class MonteCarloViewer:
    """GUI application for viewing Monte Carlo experiment data."""

    def __init__(self, root):
        """Initialize the GUI application.

        Args:
            root: tkinter root window
        """
        self.root = root
        self.root.title("Monte Carlo Data Viewer")
        self.root.geometry("1200x800")

        # Data storage
        self.current_dir = None
        self.data = {}  # {controller_name: {trajs_data, metrics}}
        self.initial_states = None

        # Color scheme (matching paper plots)
        self.colors = {
            'nmpc': '#005293',      # TUM blue
            'mpc': '#005293',       # TUM blue
            'fmpc': '#64A0C8',      # TUM light blue
            'fmpc_ext': '#64A0C8',  # TUM light blue
            'fmpc_socp': '#E37222'  # TUM orange
        }

        # State names for 2D quadrotor
        self.state_names = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
        self.state_labels = ['x [m]', r'$\dot{x}$ [m/s]', 'z [m]', r'$\dot{z}$ [m/s]',
                            r'$\theta$ [rad]', r'$\dot{\theta}$ [rad/s]']
        self.input_names = ['Thrust', 'Angle']
        self.input_labels = [r'$T_c$ [N]', r'$\theta_c$ [rad]']

        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Top frame - Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N))

        # Directory selection
        ttk.Label(control_frame, text="Run Directory:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dir_label = ttk.Label(control_frame, text="No directory selected",
                                   foreground="gray", width=60, anchor=tk.W)
        self.dir_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(control_frame, text="Browse...",
                  command=self._browse_directory).grid(row=0, column=2, padx=5)

        # Controller selection
        ttk.Label(control_frame, text="Controller:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.controller_var = tk.StringVar()
        self.controller_combo = ttk.Combobox(control_frame, textvariable=self.controller_var,
                                            state='disabled', width=20)
        self.controller_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.controller_combo.bind('<<ComboboxSelected>>', self._on_controller_change)

        # Trial selection
        ttk.Label(control_frame, text="Trial:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.trial_var = tk.StringVar()
        self.trial_combo = ttk.Combobox(control_frame, textvariable=self.trial_var,
                                       state='disabled', width=20)
        self.trial_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.trial_combo.bind('<<ComboboxSelected>>', self._update_plot)

        # Plot type selection
        ttk.Label(control_frame, text="Plot Type:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.plot_type_var = tk.StringVar(value='states')
        plot_type_frame = ttk.Frame(control_frame)
        plot_type_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(plot_type_frame, text="States", variable=self.plot_type_var,
                       value='states', command=self._update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(plot_type_frame, text="Inputs", variable=self.plot_type_var,
                       value='inputs', command=self._update_plot).pack(side=tk.LEFT, padx=5)

        # Add a separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=4, column=0,
                                                                columnspan=3, sticky=(tk.W, tk.E),
                                                                pady=10)

        # Info frame for displaying trial info
        self.info_frame = ttk.LabelFrame(control_frame, text="Trial Information", padding="5")
        self.info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.info_text = tk.Text(self.info_frame, height=4, width=80, state='disabled')
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Plot frame
        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.S, tk.N))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _browse_directory(self):
        """Open directory browser to select a Monte Carlo run directory."""
        initial_dir = '/home/ahall/Documents/UofT/papers/mimo_fmpc_socp_gp_ecc/mimo_fmpc_socp/runs_paper/monte_carlo_results'
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser('~')

        directory = filedialog.askdirectory(
            title="Select Monte Carlo Run Directory",
            initialdir=initial_dir
        )

        if directory:
            self._load_directory(directory)

    def _load_directory(self, directory):
        """Load data from the selected directory.

        Args:
            directory (str): Path to the Monte Carlo run directory
        """
        try:
            # Check if directory contains expected files
            pkl_files = [f for f in os.listdir(directory) if f.endswith('_trials.pkl')]

            if not pkl_files:
                messagebox.showerror("Error",
                    "No trial data files (*_trials.pkl) found in selected directory.")
                return

            # Load data for each controller
            self.data = {}
            controller_names = []

            for pkl_file in pkl_files:
                controller_name = pkl_file.replace('_trials.pkl', '')
                filepath = os.path.join(directory, pkl_file)

                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        self.data[controller_name] = data
                        controller_names.append(controller_name)
                except ModuleNotFoundError as e:
                    messagebox.showerror("Import Error",
                        f"Failed to load {pkl_file}:\n{str(e)}\n\n"
                        f"Make sure you're running in the 'safe' conda environment:\n"
                        f"conda activate safe")
                    return

            # Try to load initial states
            initial_states_file = os.path.join(directory, 'initial_states.pkl')
            if os.path.exists(initial_states_file):
                with open(initial_states_file, 'rb') as f:
                    init_data = pickle.load(f)
                    self.initial_states = init_data['initial_states']

            # Update UI
            self.current_dir = directory
            self.dir_label.config(text=os.path.basename(directory), foreground="black")

            # Populate controller dropdown
            self.controller_combo['values'] = sorted(controller_names)
            self.controller_combo['state'] = 'readonly'
            if controller_names:
                self.controller_combo.current(0)
                self._on_controller_change(None)

            messagebox.showinfo("Success",
                f"Loaded data for {len(controller_names)} controller(s)")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def _on_controller_change(self, event):
        """Handle controller selection change."""
        controller = self.controller_var.get()
        if not controller or controller not in self.data:
            return

        # Get number of trials
        trajs_data = self.data[controller]['trajs_data']
        n_trials = len(trajs_data['obs'])

        # Populate trial dropdown
        trial_options = [f"Trial {i+1}" for i in range(n_trials)]
        self.trial_combo['values'] = trial_options
        self.trial_combo['state'] = 'readonly'
        if n_trials > 0:
            self.trial_combo.current(0)
            self._update_plot(None)

    def _update_plot(self, event):
        """Update the plot with selected data."""
        controller = self.controller_var.get()
        trial_str = self.trial_var.get()

        if not controller or not trial_str:
            return

        # Extract trial index
        trial_idx = int(trial_str.split()[1]) - 1

        # Clear previous plot
        self.fig.clear()

        # Get data
        trajs_data = self.data[controller]['trajs_data']

        # Update info text
        self._update_info(controller, trial_idx, trajs_data)

        # Plot based on selection
        plot_type = self.plot_type_var.get()
        if plot_type == 'states':
            self._plot_states(trajs_data, trial_idx, controller)
        else:
            self._plot_inputs(trajs_data, trial_idx, controller)

        self.canvas.draw()

    def _update_info(self, controller, trial_idx, trajs_data):
        """Update the information text box.

        Args:
            controller (str): Controller name
            trial_idx (int): Trial index
            trajs_data (dict): Trajectory data
        """
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)

        info_lines = []
        info_lines.append(f"Controller: {controller.upper()}")
        info_lines.append(f"Trial: {trial_idx + 1}")

        # Get trajectory length
        n_timesteps = len(trajs_data['obs'][trial_idx])
        info_lines.append(f"Timesteps: {n_timesteps}")

        # Get initial state if available
        if self.initial_states is not None and trial_idx < len(self.initial_states):
            init_state = self.initial_states[trial_idx]
            info_lines.append(f"Initial state: [{', '.join([f'{x:.3f}' for x in init_state])}]")

        self.info_text.insert('1.0', '\n'.join(info_lines))
        self.info_text.config(state='disabled')

    def _plot_states(self, trajs_data, trial_idx, controller):
        """Plot states for a single trial.

        Args:
            trajs_data (dict): Trajectory data
            trial_idx (int): Trial index
            controller (str): Controller name
        """
        obs = trajs_data['obs'][trial_idx]  # Shape: (n_timesteps, 6)
        n_timesteps = len(obs)

        # Assume 50 Hz control frequency (can be made configurable)
        dt = 0.02
        time = np.arange(n_timesteps) * dt

        # Get controller color
        color = self.colors.get(controller, '#000000')

        # Create subplots (3x2 grid for 6 states)
        axes = self.fig.subplots(3, 2)
        axes = axes.flatten()

        for i, (ax, state_label) in enumerate(zip(axes, self.state_labels)):
            ax.plot(time, obs[:, i], color=color, linewidth=2, alpha=0.8)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(state_label)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'State: {self.state_names[i]}')

        self.fig.suptitle(f'{controller.upper()} - Trial {trial_idx + 1} - States',
                         fontsize=14, fontweight='bold')
        self.fig.tight_layout()

    def _plot_inputs(self, trajs_data, trial_idx, controller):
        """Plot inputs for a single trial.

        Args:
            trajs_data (dict): Trajectory data
            trial_idx (int): Trial index
            controller (str): Controller name
        """
        action = trajs_data['action'][trial_idx]  # Shape: (n_timesteps, 2)
        n_timesteps = len(action)

        # Assume 50 Hz control frequency (can be made configurable)
        dt = 0.02
        time = np.arange(n_timesteps) * dt

        # Get controller color
        color = self.colors.get(controller, '#000000')

        # Create subplots (2x1 grid for 2 inputs)
        axes = self.fig.subplots(2, 1)

        for i, (ax, input_label) in enumerate(zip(axes, self.input_labels)):
            ax.plot(time, action[:, i], color=color, linewidth=2, alpha=0.8)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(input_label)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Input: {self.input_names[i]}')

        self.fig.suptitle(f'{controller.upper()} - Trial {trial_idx + 1} - Inputs',
                         fontsize=14, fontweight='bold')
        self.fig.tight_layout()


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = MonteCarloViewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
