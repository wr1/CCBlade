import matplotlib.pyplot as plt


def plot_power_thrust(outputs, x=None, xlabel=''):
    """Plot power and thrust curves."""
    if x is None:
        x = list(range(len(outputs.P)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(x, outputs.P, label='Power')
    ax1.set_ylabel('Power (W)')
    ax1.grid()
    ax1.legend()

    ax2.plot(x, outputs.T, label='Thrust')
    ax2.set_ylabel('Thrust (N)')
    ax2.set_xlabel(xlabel)
    ax2.grid()
    ax2.legend()

    return fig


def plot_omega_pitch(Uinf, Omega, pitch):
    """Plot omega and pitch over Uinf."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(Uinf, Omega, label='Omega (RPM)')
    plt.plot(Uinf, pitch, label='Pitch (deg)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Omega / Pitch')
    plt.legend()
    plt.grid()
    return fig
