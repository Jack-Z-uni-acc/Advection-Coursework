###############################################################################
# Date Revised: 20 Oct 2026
# Authors: Hilary Weller <h.weller@reading.ac.uk>, Jack Zanker <j.zanker26@imperial.ac.uk>
# Description: Basic tests of advection schemes
###############################################################################
def main():

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from advectionHelpers import cosBell_01_05, indicator_01_05, l2error
    from schemes import FTBS, CTCS, FTCS, Spectral

    # First make sure that the plots folder exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    def compareSchemes(init, schemes, print_courant=True, outFile=None, plot=False, return_error=False,
                               nx = 40, nt = 10, u = 1, endTime = 0.4,
                               xLim = [0,1], yLim = [None,None]):
        '''
        Authors: Hilary Weller <h.weller@reading.ac.uk>, Jack Zanker <j.zanker26@imperial.ac.uk>
        Compares snapshots of results of advection schemes using periodic boundaries
        and records numerical errors
        Parameters:
            init (callable): Function to set the initial conditions.
            schemes (list of callable): List of functions to evaluate the schemes.
            print_courant (bool): Prints the Courant number if True
            outFile (string): The name of the output file.
            plot (bool): Plots solution snapshots if True
            return_error (bool): Returns the numerical error resulting from each 
            scheme if True
            nx (int): The number of points in space
            nt (int): The number of time steps.
            u (float): The wind speed
            endTime (float) : The final time
            xLim (tuple): The minimum (xLim[0]) and maximim (xLim[1]) x-values of
                          the domain. xLim[1] is not part of the domain when periodic
            yLim (tuple): The minimum (yLim[0]) and maximum (yLim[1]) y values of
                          the graph. If [None,None] is specified then default limits
                          are used.
        Returns:
            None if return_error is False
            errors (1darray): 1d array of l2 error at endTime incurred by each scheme
                              if return_error is True.
        '''
        # Derived parameters
        dx = (xLim[1] - xLim[0])/nx
        x = np.arange(xLim[0], xLim[1], dx)
        dt = endTime/nt
        c = nx*u*endTime/(nt*(xLim[1] - xLim[0])) # Courant number, dt*u/dx
        params = (c, dt, xLim)
        if print_courant:
            print('Courant number =', c)
        
        # Set the initial conditions
        phi0 = init(x)
        
        # Colours for each line
        cols = ['blue', 'red', 'green', 'cyan', 'magenta', 'brown', 'purple',
                'pink', 'orange', 'gray', 'orange']
        
        # Find and plot the analytic solution
        phiA = init((x - u*endTime)%(xLim[1] - xLim[0]))
        if plot:
            plt.plot(x, phiA, 'k', label = "Analytic", linewidth=2)
        
        if return_error:
            errors = []

        # Calculate the results of all of the schemes and plot / record error
        for i in range(len(schemes)):
            # The solution, phi, is stored as a list of arrays at different time
            # steps, phiLevels. Initially just the initial conditions. The latest
            # time level array is always phiLevels[-1]
            phiLevels = [phi0.copy()]
            for _ in range(nt):
                schemes[i](phiLevels, params)
            if plot:
                plt.plot(x, phiLevels[-1], cols[i], label = schemes[i].__name__)
            if return_error:
                err = l2error(phiLevels[-1], phiA)
                errors.append(err)
        
        # Plot solution snapshots
        if plot:
            plt.xlim(xLim)
            plt.ylim(yLim)
            plt.xlabel('x')
            plt.ylabel(r'$\phi$')
            plt.legend()
            plt.savefig("plots/"+outFile)
            plt.close()

        # Return the l2 error for each scheme
        if return_error:
            errors = np.array(errors)
            return errors

    # List of functions to do the advection
    schemes = [[FTBS, CTCS, FTCS], [FTBS, CTCS, Spectral]]
    # List of initial conditions
    inits = [cosBell_01_05, indicator_01_05]
    # Filenames for storage
    filenames = ['basicSchemesCos.pdf', 'basicSchemesSquare.pdf', 'spectralCos.pdf', 'spectralSquare.pdf']

    # Solve the advection equation for each scheme and initial condition
    file_idx = 0
    print_courant = False

    for schemelist in schemes:
        for init in inits:
            if file_idx == len(filenames) - 1:
                print_courant = True
            compareSchemes(init, schemelist, outFile=filenames[file_idx], plot=True,
                           nx = 40, nt = 20, u = 1, endTime = 0.2, xLim = [0,1], print_courant=print_courant)
            file_idx += 1

    def plotErrors(nx_range, nt=20, xLim=[0,1], endTime=0.2):
        '''
        Author: Jack Zanker <j.zanker@imperial.ac.uk>
        Plots the l2 error at terminal time for numerical advection schemes relative to 
        spatial discretisation stepsizes.
        Parameters:
            nx_range (1darray): Array of integer spatial discretiation mesh sizes to test over.
            nt (int): Number of timesteps.
            xLim (tuple): The minimum (xLim[0]) and maximim (xLim[1]) x-values of
              the domain. xLim[1] is not part of the domain when periodic
            endTime (float) : The final time
        Returns:
            None
        '''

        # Setup parameters to test over all schemes and a range of discretisation steps
        all_schemes = [FTBS, CTCS, FTCS, Spectral]
        dx_range = 1 / nx_range
        return_error = True
        print_courant = False
        init_errors = []
        error_filenames = ['FDcosError.pdf', 'SpectralcosError.pdf', 'squareError.pdf', 'SpectralsquareError.pdf']
        idx = 0

        # Create separate plots for each initial condition comparing finitie difference schemes
        for init in inits:
            # Error as a function of spatial discretisation step
            nx_errors = []
            for nx in nx_range:
                error = compareSchemes(init, all_schemes, nx=nx, nt=20, u=1, endTime=endTime, xLim=xLim,
                             print_courant=print_courant, return_error=True)
                nx_errors.append(error)
            nx_errors = np.array(nx_errors).T
            # Produce and save error plots
            for i in range(len(all_schemes) - 1):
                plt.plot(dx_range, nx_errors[i], label=all_schemes[i].__name__)
            plt.xlabel('Stepsize')
            plt.ylabel('l2 error')
            plt.yscale('linear')
            plt.legend()
            plt.savefig('plots/'+error_filenames[idx])
            plt.close()
            idx +=1
            # Plot the error for the spectral method
            plt.plot(dx_range, nx_errors[-1], label='Spectral')
            plt.xlabel('Stepsize')
            plt.ylabel('l2 error')
            plt.yscale('linear')
            plt.legend()
            plt.savefig('plots/'+error_filenames[idx])
            plt.close()
            idx+=1

    # Parameters to test over
    nx_range = 2**np.arange(2, 10)
    plotErrors(nx_range)

# Run script

if __name__ == '__main__':
    main()





































