import matplotlib.pyplot as plt

from datetime import datetime,timedelta
import numpy as np
from netCDF4 import Dataset,date2num,num2date
import pdb,os,sys
import os.path
#import shutil
#from shutil import copyfile,move
import xarray
import glob
import cftime
import matplotlib
#from matplotlib.transforms import Bbox

sys.path.append('/net/home/h01/hadjt/workspace/python3/')
#sys.path.append('/home/d05/hadjt/scripts/python/')

#from matplotlib.backend_bases import MouseButton

from NEMO_nc_slevel_viewer_lib import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region,interp1dmat_wgt, interp1dmat_create_weight, nearbed_index,extract_nb,mask_stats,load_nearbed_index,pea_TS

# my tools to change the colorbar limits, mainly to set to the 5th and 95th percentile of the plotted data.
#from python3_plotting_function import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region
# efficient way to extract z levels from s level data (effectively linear interpolation)
#from nemo_forcings_functions import interp1dmat_wgt, interp1dmat_create_weight
# indices to quickly extract near bed values.
#from nemo_forcings_functions import  nearbed_index,extract_nb,mask_stats,load_nearbed_index
# Allow the amm15 grid to be unrotated, to allow efficient coversion between lon, lats to ii,jj's. 
from rotated_pole_grid import rotated_grid_from_amm15,rotated_grid_to_amm15, reduce_rotamm15_grid

#scp -pr  NEMO_nc_slevel_viewer*.py ../rotated_pole_grid*  hadjt@xcel00:/home/d05/hadjt/scripts/python/.

letter_mat = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


import argparse
import textwrap

import socket
computername = socket.gethostname()
comp = 'linux'
if computername in ['xcel00','xcfl00']: comp = 'hpc'

if comp == 'linux': sys.path.append('/home/h01/hadjt/workspace/python3/')
if computername in ['xcel00','xcfl00']: sys.path.append('/home/d05/hadjt/scripts/python/')


import warnings
warnings.filterwarnings("ignore")

def load_nc_dims(tmp_data):
    x_dim = 'x'
    y_dim = 'y'
    z_dim = 'deptht'
    t_dim = 'time_counter'
    #pdb.set_trace()
    
    nc_dims = [ss for ss in tmp_data._dims.keys()]

    poss_zdims = ['deptht','depthu','depthv']
    poss_tdims = ['time_counter','time']
    poss_xdims = ['x','X']
    poss_ydims = ['y','Y']

    if x_dim not in nc_dims: x_dim = [i for i in nc_dims if i in poss_xdims][0]
    if y_dim not in nc_dims: y_dim = [i for i in nc_dims if i in poss_ydims][0]
    if z_dim not in nc_dims: z_dim = [i for i in nc_dims if i in poss_zdims][0]
    if t_dim not in nc_dims: t_dim = [i for i in nc_dims if i in poss_tdims][0]
    return x_dim, y_dim, z_dim,t_dim


def load_nc_var_name_list(tmp_data,x_dim, y_dim, z_dim,t_dim):

    # what are the4d variable names, and how many are there?
    #var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 4])
    var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim, z_dim,y_dim, x_dim)])
    nvar4d = var_4d_mat.size
    #pdb.set_trace()
    #var_3d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 3])
    var_3d_mat = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim, y_dim, x_dim)])
    nvar3d = var_3d_mat.size

    var_mat = np.append(var_4d_mat, var_3d_mat)
    nvar = var_mat.size


    var_dim = {}
    for vi,var_dat in enumerate(var_4d_mat): var_dim[var_dat] = 4
    for vi,var_dat in enumerate(var_3d_mat): var_dim[var_dat] = 3

    return var_4d_mat, var_3d_mat, var_mat, nvar4d, nvar3d, nvar, var_dim



def nemo_slice_zlev(fname_lst, subtracted_flist = None,var = None,config = 'amm7', thin = 1,
    zlim_max = None,xlim = None, ylim = None, tlim = None, clim = None,
    ii = None, jj = None, ti = None, zz = None,
    clim_sym = None, use_cmocean = False,
    U_flist = None,V_flist = None,
    fig_dir = '/home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer/tmpfigs',
    fig_lab = 'figs',fig_cutout = True,
    verbose_debugging = False):


    if verbose_debugging:
        print('======================================================')
        print('======================================================')
        print('=== Debugging printouts: verbose_debugging = True  ===')
        print('======================================================')
        print('======================================================')


    #Default variable for U and V flist
    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'





    if use_cmocean:
        
        import cmocean
        # default color map to use
        base_cmap = None
        diff_cmap = cmocean.cm.balance
    else:
        base_cmap = None
        #diff_cmap = matplotlib.cm.seismic
        diff_cmap = matplotlib.cm.coolwarm
    curr_cmap = base_cmap

    if clim_sym is None: clim_sym = False

    # default initial indices
    if ii is None: ii = 10
    if jj is None: jj = 10
    if ti is None: ti = 0
    if zz is None: zz = 0
    if zz == 0: zi = 0

    # Set mode: Click or Loop
    mode = 'Click'
    loop_sleep = 0.01

    load_2nd_files = False
    # repeat if comparing two time series. 
    if subtracted_flist is not None:
        
        load_2nd_files = True



    # if a secondary data set, give ability to change data sets. 
    secdataset_proc = 'Dataset 1'
    secdataset_proc_list = ['Dataset 1', 'Dataset 2', 'Dat2-Dat1']
    if load_2nd_files:
        secdataset_proc = 'Dat2-Dat1'



    #config version specific info - mainly grid, and lat/lon info
    if config.upper() == 'AMM7':
        # depth grid file
        if comp == 'hpc': 
            amm7_mesh_file = '/data/d02/frpk/amm15ps45mesh/amm7/amm7.mesh_mask.nc'
            nemo_nb_i_filename = '/data/d05/hadjt/reffiles/NEMO_nc_viewer/nemo_nb_i_CMEMS_BGC_Reanalysis_14112017.nc'
        else:
            amm7_mesh_file = '/data/cr1/hadjt/data/reffiles/SSF/amm7.mesh_mask.nc'
            nemo_nb_i_filename = '/home/h01/hadjt/Work/Programming/Scripts/reffiles/nemo_nb_i_CMEMS_BGC_Reanalysis_14112017.nc'
        rootgrp_gdept = Dataset(amm7_mesh_file, 'r', format='NETCDF4')
        # depth grid variable name
        zss = 'gdept'

        #grid lat lon
        lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
        lat = np.arange(40.066669,65+1/15.,1/15.)
        nbind,tmask = load_nearbed_index(nemo_nb_i_filename = nemo_nb_i_filename)
        z_meth_default = 'z_slice'
        z_meth = z_meth_default
        
    elif config.upper() == 'AMM15':

        # depth grid file
        if comp == 'hpc':
            amm15_mesh_file = '/data/d02/frpk/amm15ps45mesh/amm15.mesh_mask.nc'
            nemo_nb_i_filename = '/data/d05/hadjt/reffiles/NEMO_nc_viewer/nemo_nb_i_OpSys_AMM15_NEMO36.nc'
        else:
            amm15_mesh_file = '/data/cr1/hadjt/data/reffiles/SSF/amm15.mesh_mask.nc'
            nemo_nb_i_filename = '/data/cr1/hadjt/data/reffiles/SSF/nemo_nb_i_OpSys_AMM15_NEMO36.nc'
        rootgrp_gdept = Dataset(amm15_mesh_file, 'r', format='NETCDF4')
        # depth grid variable name
        zss = 'gdept_0'
        
        # grid lat lon rotation information
        lon_rotamm15,lat_rotamm15 = reduce_rotamm15_grid()

        dlon_rotamm15 = (np.diff(lon_rotamm15)).mean()
        dlat_rotamm15 = (np.diff(lat_rotamm15)).mean()
        nlon_rotamm15 = lon_rotamm15.size
        nlat_rotamm15 = lat_rotamm15.size



        #moo ls moose:/devfc/rosie_OS45_LBC_amm15_control/field.nc.file/prodm_op_am-dm.gridT_20220824_00.-36.nc
        #nbind_tmp,tmask_tmp = nearbed_index('/scratch/hadjt/SSF/LBC/amm15/OS45_LBC_amm15_control/prodm_op_am-dm.gridT_20220824_00.-36.nc', 'votemper',nemo_nb_i_filename = '/data/cr1/hadjt/data/reffiles/SSF/nemo_nb_i_OpSys_AMM15_NEMO36.nc')

        #nbind_tmp,tmask_tmp = nearbed_index('/scratch/orca12/g18trial/control/prodm_op_gf-dm.gridT_20211203_00.-36.nc', 'votemper',nemo_nb_i_filename = '/data/cr1/hadjt/data/reffiles/SSF/nemo_nb_i_OpSys_GULF18_NEMO36.nc')

        nbind,tmask = load_nearbed_index(nemo_nb_i_filename = nemo_nb_i_filename)
        z_meth_default = 'z_slice'
        z_meth = z_meth_default


    elif config.upper() == 'GULF18':
        # depth grid file
        if comp == 'hpc': 
            print('GULF18 files not copeid onto Cray')
            #amm7_mesh_file = '/data/d02/frpk/amm15ps45mesh/amm7/amm7.mesh_mask.nc'
            #nemo_nb_i_filename = '/data/d05/hadjt/reffiles/NEMO_nc_viewer/nemo_nb_i_CMEMS_BGC_Reanalysis_14112017.nc'
        else:
            GULF18_mesh_file = '/data/cr1/hadjt/data/reffiles/SSF/mesh_mask_gulf18_ps45.nc'
            nemo_nb_i_filename = '/data/cr1/hadjt/data/reffiles/SSF/nemo_nb_i_OpSys_GULF18_NEMO36.nc'

   
        rootgrp_gdept = Dataset(GULF18_mesh_file, 'r', format='NETCDF4')
        # depth grid variable name
        zss = 'gdept'

        #grid lat lon
        #pdb.set_trace()
        lon = rootgrp_gdept.variables['glamt'][:,0,:].ravel()
        lat = rootgrp_gdept.variables['gphit'][:,:,0].ravel()
        nbind,tmask = load_nearbed_index(nemo_nb_i_filename = nemo_nb_i_filename)
        z_meth_default = 'z_slice'
        z_meth = z_meth_default





    elif config.upper() in ['ORCA025','ORCA025EXT']:
        #add grid file (with gdept_0)
        #need to be able to quickly find ii and jj from lon, lat... 
        #add a function to do this based on grid geometry, rather than a (slow) search for closest grid cell.

        orca025ext_mesh_file = '/data/cr1/hadjt/data/reffiles/ORCA/ORCA025ext/mesh_mask_orca025ext.nc'
        if comp == 'hpc': orca025ext_mesh_file = '/projects/ofrd/NEMO/ancil/orca025ext/mesh_mask_orca025ext.nc'
        rootgrp_gdept = Dataset(orca025ext_mesh_file, 'r', format='NETCDF4')
        zss = 'gdept_0'


        z_meth_default = 'z_index'
        z_meth = z_meth_default


        #pdb.set_trace()
    elif config.upper() == 'ORCA12':
        #add grid file (with gdept_0)
        #need to be able to quickly find ii and jj from lon, lat... 
        #add a function to do this based on grid geometry, rather than a (slow) search for closest grid cell.
        z_meth_default = 'z_index'
        z_meth = z_meth_default
        pdb.set_trace()
    elif config.upper() == 'ORCA1':
        #add grid file (with gdept_0)
        #need to be able to quickly find ii and jj from lon, lat... 
        #add a function to do this based on grid geometry, rather than a (slow) search for closest grid cell.
        z_meth_default = 'z_index'
        z_meth = z_meth_default
        pdb.set_trace()
    else:
        print('config not supported')
        pdb.set_trace()
    

    # open file list with xarray
    tmp_data = xarray.open_mfdataset(fname_lst ,combine='by_coords') # , decode_cf=False)

    #Add batoclinic velocity magnitude
    UV_vec = False
    if (U_flist is not None) & (V_flist is not None):
        UV_vec = True
        tmp_data_U = xarray.open_mfdataset(U_flist ,combine='by_coords') # , decode_cf=False)
        tmp_data_V = xarray.open_mfdataset(V_flist ,combine='by_coords') # , decode_cf=False)



    # load nav_lat and nav_lon
    if config.upper() in ['ORCA025','ORCA025EXT']: 
        nav_lon = np.ma.masked_invalid(rootgrp_gdept.variables['glamt'][0])
        nav_lat = np.ma.masked_invalid(rootgrp_gdept.variables['gphit'][0])

        fixed_nav_lon = nav_lon.copy()
        for i, start in enumerate(np.argmax(np.abs(np.diff(nav_lon)) > 180, axis=1)):            fixed_nav_lon[i, start+1:] += 360
        fixed_nav_lon -=360
        fixed_nav_lon[fixed_nav_lon<-287.25] +=360
        fixed_nav_lon[fixed_nav_lon>73] -=360
        #pdb.set_trace()
        nav_lon = fixed_nav_lon.copy()

        #pdb.set_trace()

        nav_lat = np.ma.array(nav_lat[::thin,::thin])
        nav_lon = np.ma.array(nav_lon[::thin,::thin])
        
        
    else:
        nav_lat = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
        nav_lon = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())
        

    
    deriv_var = []
    x_dim, y_dim, z_dim, t_dim = load_nc_dims(tmp_data) #  find the names of the x, y, z and t dimensions.
    var_4d_mat, var_3d_mat, var_mat, nvar4d, nvar3d, nvar, var_dim = load_nc_var_name_list(tmp_data, x_dim, y_dim, z_dim,t_dim)# find the variable names in the nc file
    var_grid = {}
    for ss in var_mat: var_grid[ss] = 'T'


    if UV_vec == True:
        
        U_x_dim, U_y_dim, U_z_dim, U_t_dim  = load_nc_dims(tmp_data_U) #  find the names of the x, y, z and t dimensions.
        U_var_names = load_nc_var_name_list(tmp_data_U, U_x_dim, U_y_dim, U_z_dim,U_t_dim)# find the variable names in the nc file # var_4d_mat, var_3d_mat, var_mat, nvar4d, nvar3d, nvar, var_dim = 
        U_var_4d_mat, U_var_3d_mat, U_var_mat, U_var_dim = U_var_names[0],U_var_names[1],U_var_names[2],U_var_names[6]

        V_x_dim, V_y_dim, V_z_dim, V_t_dim = load_nc_dims(tmp_data_V) #  find the names of the x, y, z and t dimensions.
        V_var_names = load_nc_var_name_list(tmp_data_V, V_x_dim, V_y_dim, V_z_dim, V_t_dim)# find the variable names in the nc file # var_4d_mat, var_3d_mat, var_mat, nvar4d, nvar3d, nvar, var_dim
        V_var_4d_mat, V_var_3d_mat, V_var_mat, V_var_dim = V_var_names[0],V_var_names[1],V_var_names[2],V_var_names[6]
        
        var_mat = np.append(np.append(var_mat, U_var_mat), V_var_mat)
        for ss in U_var_dim: var_dim[ss] = U_var_dim[ss]
        for ss in V_var_dim: var_dim[ss] = V_var_dim[ss]
        
        
        for ss in U_var_mat: var_grid[ss] = 'U'
        for ss in V_var_mat: var_grid[ss] = 'V'

        if ('vozocrtx' in var_mat) & ('vomecrty' in var_mat):
            ss = 'baroc_mag'
            var_mat = np.append(var_mat,ss)
            var_dim[ss] = 4
            var_grid[ss] = 'UV'
            deriv_var.append(ss)


            curr_tmp_data_U = tmp_data_U
            curr_tmp_data_V = tmp_data_V

            if load_2nd_files: curr_tmp_data_diff_U = tmp_data_diff_U
            if load_2nd_files: curr_tmp_data_diff_V = tmp_data_diff_V
            
        #pdb.set_trace()


    add_PEA = False
    if ('votemper' in var_mat) & ('vosaline' in var_mat):
        add_PEA = True
        ss = 'pea'
        var_mat = np.append(var_mat,ss)
        var_dim[ss] = 3
        var_grid[ss] = 'T'
        deriv_var.append(ss)

    

    #pdb.set_trace() 

    if var is None: var = 'votemper'
    if var not in var_mat: var = var_mat[0]



    # extract time information from xarray.
    # needs to work for gregorian and 360 day calendars.
    # needs to work for as x values in a plot, or pcolormesh
    # needs work, xarray time is tricky
    nctime = tmp_data.variables['time_counter']

    '''
    if 'calendar' in nctime[0].attrs.keys():
        nc_calendar = nctime[0].attrs['calendar']
    else:
        nc_calendar = 'gregorian'
    pdb.set_trace()
    '''
    '''
    if comp == 'hpc':
        #pdb.set_trace()
        rootgrp_hpc_time = Dataset(fname_lst[0], 'r', format='NETCDF4')
        nc_time_origin = rootgrp_hpc_time.variables['time_counter'].time_origin
        rootgrp_hpc_time.close()
    else:        
        nc_time_origin = nctime[0].attrs['time_origin']
    '''
    rootgrp_hpc_time = Dataset(fname_lst[0], 'r', format='NETCDF4')
    nc_time_origin = rootgrp_hpc_time.variables['time_counter'].time_origin
    rootgrp_hpc_time.close()
        
    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    if type(np.array(nctime)[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
        nctime_calendar_type = '360'
    else:
        nctime_calendar_type = 'greg'


    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    #if type(np.array(nctime)[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
    if  nctime_calendar_type == '360':
        # if 360 days

        time_datetime_since_1970 = [ss.year + (ss.month-1)/12 + (ss.day-1)/360 for ss in np.array(nctime)]   
        time_datetime = time_datetime_since_1970
    else:
        # if gregorian

        
        sec_since_origin = [float(ii.data - np.datetime64(nc_time_origin))/1e9 for ii in nctime]
        time_datetime_cft = num2date(sec_since_origin,units = 'seconds since ' + nc_time_origin,calendar = 'gregorian') #nctime.calendar)

        time_datetime = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft])
        time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])


    ntime = time_datetime_since_1970.size

    #pdb.set_trace()
    # repeat if comparing two time series. 
    if subtracted_flist is not None:
        
        clim_sym = True
        tmp_data_diff = xarray.open_mfdataset(subtracted_flist ,combine='by_coords' )
        #pdb.set_trace()
        nav_lat_diff = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
        nav_lon_diff = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())

        nctime_diff = tmp_data_diff.variables['time_counter']
        #nc_time_origin_diff = nctime_diff[0].attrs['time_origin']


        rootgrp_hpc_time = Dataset(subtracted_flist[0], 'r', format='NETCDF4')
        nc_time_origin_diff = rootgrp_hpc_time.variables['time_counter'].time_origin
        rootgrp_hpc_time.close()

        sec_since_origin_diff = [float(ii.data - np.datetime64(nc_time_origin_diff))/1e9 for ii in nctime_diff]
        time_datetime_cft_diff = num2date(sec_since_origin_diff,units = 'seconds since ' + nc_time_origin_diff,calendar = 'gregorian') #nctime.calendar)

        time_datetime_diff = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft_diff])
        time_datetime_since_1970_diff = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime_diff])
        ntime_diff = time_datetime_since_1970_diff.size
        
        # check both filessets have the same times
        if ntime_diff != ntime:     
            print('Diff Times have different number of files')
            pdb.set_trace() 
        else:
            if (time_datetime_since_1970_diff != time_datetime_since_1970).any():   
                print('Diff Times dont match')
                pdb.set_trace()
        if (nav_lat != nav_lat_diff).any():
            print('Diff nav_lat_diff dont match')
            pdb.set_trace()
        if (nav_lon != nav_lon_diff).any():
            print('Diff nav_lon_diff dont match')
            pdb.set_trace()
        # use a difference colormap if comparing files
        curr_cmap = diff_cmap


    # set up figure.
    #   set up default figure, and then and and delete plots when you change indices.
    #   change indices with mouse click, detected with ginput
    #   ginput only works on one axes, so add a hidden fill screen axes, and then convert figure indices to an axes, and then using axes position and x/ylims into axes index. 
    #   create boxes with variable names as buttons to change variables. 
    climnorm = None # matplotlib.colors.LogNorm(0.005,0.1)
    
    print('Creating Figure')

    ax = []
    pax = []

    fig = plt.figure()
    fig.suptitle('Interactive figure, double click to select, click outside axes to quit. Select lat/lon in a); lon in b); lat  in c); depth in d) and time in e)', fontsize=14)
    fig.set_figheight(12)
    fig.set_figwidth(18)
    if nvar <18:
        plt.subplots_adjust(top=0.9,bottom=0.11,left=0.08,right=0.9,hspace=0.2,wspace=0.135)
    else:
        plt.subplots_adjust(top=0.9,bottom=0.11,left=0.15,right=0.9,hspace=0.2,wspace=0.135)
    # add axes
    ax.append(plt.subplot(1,2,1))
    ax.append(plt.subplot(4,2,2))
    ax.append(plt.subplot(4,2,4))
    ax.append(plt.subplot(4,2,6))
    ax.append(plt.subplot(4,2,8))



    labi,labj = 0.05, 0.95
    for ai,tmpax in enumerate(ax): tmpax.text(labi,labj,'%s)'%letter_mat[ai], transform=tmpax.transAxes, ha = 'left', va = 'top', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none'))

    #flip depth axes
    for tmpax in ax[1:]: tmpax.invert_yaxis()
    #use log depth scale, setiched off as often causes problems (clashes with hidden axes etc).
    #for tmpax in ax[1:]: tmpax.set_yscale('log')

    # add hidden fill screen axes 
    clickax = fig.add_axes([0,0,1,1], frameon=False)
    clickax.axis('off')


    if verbose_debugging: print('Created figure', datetime.now())

    
    #add "buttons"
    but_x0 = 0.01
    but_x1 = 0.06
    func_but_x1 = 0.99
    func_but_x0 = 0.94
    func_but_dx1 = func_but_x1 -func_but_x0 
    but_dy = 0.04
    but_ysp = 0.01 


    but_extent = {}
    but_line_han,but_text_han = {},{}
    for vi,var_dat in enumerate(var_mat): 
        tmpcol = 'k'
        if var_dim[var_dat] == 3: tmpcol = 'darkgreen'
        if var_grid[var_dat] != 'T': tmpcol = 'gold'
        if var_dat in deriv_var: tmpcol = '0.5'
        vi_num = vi
        if vi>=18:
            vi_num = vi-18

            but_x0 = 0.01 + 0.06
            but_x1 = 0.06 + 0.06
            
        #note button extends (as in position.x0,x1, y0, y1)
        but_extent[var_dat] = np.array([but_x0,but_x1,0.9 - (but_dy + vi*0.05),0.9 - (0 + vi_num*0.05)])
        #add button box
        but_line_han[var_dat] = clickax.plot([but_x0,but_x1,but_x1,but_x0,but_x0],0.9 - (np.array([0,0,but_dy,but_dy,0]) + vi_num*0.05),color = tmpcol)
        #add button names
        but_text_han[var_dat] = clickax.text((but_x0+but_x1)/2,0.9 - ((but_dy/2) + vi_num*0.05),var_dat, ha = 'center', va = 'center')


    clickax.axis([0,1,0,1])
    
    if verbose_debugging: print('Added variable boxes', datetime.now())

    mode_name_lst = ['Click','Loop']

    func_names_lst = ['Reset zoom', 'Zoom', 'Clim: Reset','Clim: Zoom','Clim: Expand','Clim: perc','Clim: normal', 'Clim: log','Surface', 'Near-Bed', 'Surface-Bed','Depth level','Save Figure', 'Quit']
    
    func_names_lst = func_names_lst + mode_name_lst

    # if a secondary data set, give ability to change data sets. 
    if load_2nd_files:
        func_names_lst = func_names_lst + secdataset_proc_list

    func_but_line_han,func_but_text_han = {},{}
    func_but_extent = {}


    mode_name_secdataset_proc_list = mode_name_lst

    if load_2nd_files: 
        mode_name_secdataset_proc_list = mode_name_secdataset_proc_list + secdataset_proc_list

    #add button box
    for vi,funcname in enumerate(func_names_lst): 

        #note button extends (as in position.x0,x1, y0, y1)
        func_but_extent[funcname] = [func_but_x0,func_but_x1,0.9 - (but_dy + vi*0.05),0.9 - (0 + vi*0.05)]


    for vi, tmp_funcname in enumerate(mode_name_secdataset_proc_list):
        func_but_extent[tmp_funcname] = [0.15 + vi*(func_but_dx1+0.01), 0.15 + vi*(func_but_dx1+0.01) + func_but_dx1, 0.025,  0.025 + but_dy]

    for vi,funcname in enumerate(func_names_lst): 

        func_but_line_han[funcname] = clickax.plot([func_but_extent[funcname][0],func_but_extent[funcname][1],func_but_extent[funcname][1],func_but_extent[funcname][0],func_but_extent[funcname][0]], [func_but_extent[funcname][2],func_but_extent[funcname][2],func_but_extent[funcname][3],func_but_extent[funcname][3],func_but_extent[funcname][2]],'k')
         #add button names
        func_but_text_han[funcname] = clickax.text((func_but_extent[funcname][0]+func_but_extent[funcname][1])/2,(func_but_extent[funcname][2]+func_but_extent[funcname][3])/2,funcname, ha = 'center', va = 'center')
    
    #pdb.set_trace()  
    
    # if a secondary data set, det default behaviour. 
    if load_2nd_files: func_but_text_han[secdataset_proc].set_color('darkgreen')


    # Set intial mode to be Click
    func_but_text_han['Click'].set_color('gold')

    # When we move to loop mode, we stop checking for button presses, 
    #   so need another way to end the loop... 
    #       could just wait till the end of the loop, but could be ages
    #   therefore see if the mouse points to the "Click" button and change the mode.
    #
    # this could be an alternate method to the plt.ginput method.
    # Therefore define a global variable "mouse_in_Click" and use the matplotlib 
    # Connect, 'motion_notify_event', on_move method:
    #   https://matplotlib.org/stable/gallery/event_handling/coords_demo.html
    #
    global mouse_in_Click
    mouse_in_Click = False

    def on_move(event):
        global mouse_in_Click
        if event.inaxes:


            if (event.xdata>func_but_extent['Click'][0]) & (event.xdata<func_but_extent['Click'][1]) & (event.ydata>func_but_extent['Click'][2]) & (event.ydata<func_but_extent['Click'][3]):
                mouse_in_Click = True
                if verbose_debugging: print('Mouse in Click',datetime.now())
            else:
                mouse_in_Click = False


    binding_id = plt.connect('motion_notify_event', on_move)




    func_but_text_han['Depth level'].set_color('r')
    func_but_text_han['Clim: normal'].set_color('b')
    but_text_han[var].set_color('r')

    if verbose_debugging: print('Added functions boxes', datetime.now())


    ###########################################################################
    # Define inner functions
    ###########################################################################



    if verbose_debugging: print('Create inner functions', datetime.now())
    def indices_from_ginput_ax(clii,cljj,thin=thin):
        
        '''
        ginput doesn't tell you which subplot you are clicking, only the position within that subplot.
        we need which axis is clicked as well as the cooridinates within that axis
        
        we therefore trick ginput to give use figure coordinate (with a dummy, invisible full figure size subplot
        in front of everything, and then use this function to turn those coordinates into the coordinates within the 
        the subplot, and the which axis/subplot it is


        '''
        sel_ii,sel_jj,sel_ti ,sel_zz = None,None,None,None
        sel_ax = None
    
        for ai,tmpax in enumerate(ax): 
            tmppos =  tmpax.get_position()
            # was click within extent
            if (clii >= tmppos.x0) & (clii <= tmppos.x1) & (cljj >= tmppos.y0) & (cljj <= tmppos.y1):
                sel_ax = ai

                #convert figure coordinate of click, into location with the axes, using data coordinates
                clxlim = np.array(tmpax.get_xlim())
                clylim = np.array(tmpax.get_ylim())
                normxloc = (clii - tmppos.x0 ) / (tmppos.x1 - tmppos.x0)
                normyloc = (cljj - tmppos.y0 ) / (tmppos.y1 - tmppos.y0)
                xlocval = normxloc*clxlim.ptp() + clxlim.min()
                ylocval = normyloc*clylim.ptp() + clylim.min()
                #print(clii,clxlim,normxloc,xlocval)
                #print(cljj,clylim,normyloc,ylocval)

                if (thin != 1):
                    if config.upper() not in ['AMM7','AMM15', 'ORCA025','ORCA025EXT']:
                        print('Thinning lon lat selection not programmed for ', config)
                        pdb.set_trace()


                # what do the local coordiantes of the click mean in terms of the data to plot.
                # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.

                # if in map, covert lon lat to ii,jj
                if ai == 0:
                    #pdb.set_trace()
                    loni,latj= xlocval,ylocval
                    if config.upper() in ['AMM7','GULF18']:
                        sel_ii = (np.abs(lon[::thin] - loni)).argmin()
                        sel_jj = (np.abs(lat[::thin] - latj)).argmin()
                    elif config.upper() == 'AMM15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        #sel_ii = np.minimum(np.maximum( np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int') ,0),nlon_rotamm15-1)
                        #sel_jj = np.minimum(np.maximum( np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int') ,0),nlat_rotamm15-1)
                        sel_ii = np.minimum(np.maximum( np.round((lon_mat_rot - lon_rotamm15[::thin].min())/(dlon_rotamm15*thin)).astype('int') ,0),nlon_rotamm15//thin-1)
                        sel_jj = np.minimum(np.maximum( np.round((lat_mat_rot - lat_rotamm15[::thin].min())/(dlat_rotamm15*thin)).astype('int') ,0),nlat_rotamm15//thin-1)
                    elif config.upper() in ['ORCA025','ORCA025EXT']:
                        #pdb.set_trace()
                        sel_dist_mat = np.sqrt((nav_lon[:,:] - loni)**2 + (nav_lat[:,:] - latj)**2 )
                        sel_jj,sel_ii = sel_dist_mat.argmin()//sel_dist_mat.shape[1], sel_dist_mat.argmin()%sel_dist_mat.shape[1]

                    else:
                        print('config not supported:', config)
                        pdb.set_trace()
                    # and reload slices, and hovmuller/time series

                elif ai in [1]: 
                    # if in ew slice, change ns slice, and hov/time series
                    loni= xlocval
                    if config.upper() == 'AMM7':
                        sel_ii = (np.abs(lon[::thin] - loni)).argmin()
                    elif config.upper() == 'AMM15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        #sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                        sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15[::thin].min())/(dlon_rotamm15*thin)).astype('int'),0),nlon_rotamm15//thin-1)
                    else:
                        print('config not supported:', config)
                        pdb.set_trace()
                    
                    
                elif ai in [2]:
                    # if in ns slice, change ew slice, and hov/time series
                    latj= xlocval
                    if config.upper() == 'AMM7':
                        sel_jj = (np.abs(lat[::thin] - latj)).argmin()
                    elif config.upper() == 'AMM15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        #sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)
                        sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15[::thin].min())/(dlat_rotamm15*thin)).astype('int'),0),nlat_rotamm15//thin-1)
                    else:
                        print('config not supported:', config)
                        pdb.set_trace()

                elif ai in [3]:
                    # if in hov/time series, change map, and slices

                    # re calculate depth values, as y scale reversed, 
                    sel_zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )


                elif ai in [4]:
                    # if in hov/time series, change map, and slices
                    sel_ti = np.abs(xlocval - time_datetime_since_1970).argmin()
                    
                else:
                    print('clicked in another axes??')
                    return
                    pdb.set_trace()


        
        return sel_ax,sel_ii,sel_jj,sel_ti,sel_zz



    def indices_from_ginput_cax(cclii,ccljj):
        '''
        I think this is no longer called

        '''
        sel_cii,sel_cjj= None,None
        sel_cax = None
    
        for cai,tmpcax in enumerate(cax): 
            #pdb.set_trace()
            tmpcpos =  tmpcax.ax.get_position()
            
            # was click within extent
            if (cclii >= tmpcpos.x0) & (cclii <= tmpcpos.x1) & (ccljj >= tmpcpos.y0) & (ccljj <= tmpcpos.y1):
                sel_cax = ai

                #convert figure coordinate of click, into location with the axes, using data coordinates
                clxlim = np.array(tmpax.get_xlim())
                clylim = np.array(tmpax.get_ylim())
                normxloc = (cclii - tmppos.x0 ) / (tmppos.x1 - tmppos.x0)
                normyloc = (ccljj - tmppos.y0 ) / (tmppos.y1 - tmppos.y0)
                sel_cii = normxloc*clxlim.ptp() + clxlim.min()
                sel_cjj = normyloc*clylim.ptp() + clylim.min()

        sel_cjj = ccljj
               
        return sel_cax,sel_cii,sel_cjj

    def reload_ew_data():
        '''
        reload the data for the E-W cross-section

        '''
        if secdataset_proc == 'Dataset 1':
            ew_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][ti,:,jj,:].load())
        elif secdataset_proc == 'Dataset 2':
            ew_slice_dat = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][ti,:,jj,:].load())
        elif secdataset_proc =='Dat2-Dat1':
            ew_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][ti,:,jj,:].load())
            ew_slice_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][ti,:,jj,:].load())

        ew_slice_x =  nav_lon[jj,:]
        ew_slice_y =  rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,jj,:]
        return ew_slice_dat,ew_slice_x, ew_slice_y

    def reload_ns_data():              
        '''
        reload the data for the N-S cross-section

        '''
        if secdataset_proc == 'Dataset 1':
            ns_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][ti,:,:,ii].load())
        elif secdataset_proc == 'Dataset 2':
            ns_slice_dat = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][ti,:,:,ii].load())
        elif secdataset_proc =='Dat2-Dat1':
            ns_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][ti,:,:,ii].load())
        ns_slice_x =  nav_lat[:,ii]
        ns_slice_y =  rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,:,ii]
        return ns_slice_dat,ns_slice_x, ns_slice_y
                    
    def reload_hov_data():                
        '''
        reload the data for the Hovmuller plot
        '''
        if secdataset_proc == 'Dataset 1':
            hov_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][:,:,jj,ii].load()).T
        elif secdataset_proc == 'Dataset 2':
            hov_dat = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][:,:,jj,ii].load()).T
        elif secdataset_proc =='Dat2-Dat1':
            hov_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,::thin,::thin][:,:,jj,ii].load()).T
        hov_x = time_datetime
        hov_y =  rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,jj,ii]
        return hov_dat,hov_x,hov_y





    def reload_ew_data_derived_var():
        if var == 'baroc_mag':
            ew_slice_dat,ew_slice_x, ew_slice_y = reload_ew_data_derived_var_baroc_mag()
        else:
            print('var not in deriv_var',var)
        return ew_slice_dat,ew_slice_x, ew_slice_y

    def reload_ns_data_derived_var():              
        if var == 'baroc_mag':
            ns_slice_dat,ns_slice_x, ns_slice_y = reload_ns_data_derived_var_baroc_mag()
        else:
            print('var not in deriv_var',var)
        return ns_slice_dat,ns_slice_x, ns_slice_y

    def reload_hov_data_derived_var():                
        if var == 'baroc_mag':
            hov_dat,hov_x,hov_y = reload_hov_data_derived_var_baroc_mag()
        else:
            print('var not in deriv_var',var)
        return hov_dat,hov_x,hov_y






    def reload_ew_data_derived_var_baroc_mag():
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        if secdataset_proc == 'Dataset 1':
            ew_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,jj,:].load())
            ew_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,jj,:].load())
        elif secdataset_proc == 'Dataset 2':
            ew_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,jj,:].load())
            ew_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,jj,:].load())
        elif secdataset_proc =='Dat2-Dat1':
            ew_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,jj,:].load())
            ew_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,jj,:].load())
            ew_slice_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[:,:,::thin,::thin][tmp_var_U][ti,:,jj,:].load())
            ew_slice_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[:,:,::thin,::thin][tmp_var_V][ti,:,jj,:].load())
        ew_slice_dat = np.sqrt(ew_slice_dat_U**2 + ew_slice_dat_V**2)
        ew_slice_x =  nav_lon[jj,:]
        ew_slice_y =  rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,jj,:]
        return ew_slice_dat,ew_slice_x, ew_slice_y

    def reload_ns_data_derived_var_baroc_mag():              
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        if secdataset_proc == 'Dataset 1':
            ns_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,:,ii].load())
        if secdataset_proc == 'Dataset 1':
            ns_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,:,ii].load())
        elif secdataset_proc =='Dat2-Dat1':
            ns_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,::thin,::thin][ti,:,:,ii].load())
            ns_slice_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,::thin,::thin][ti,:,:,ii].load())
        ns_slice_dat = np.sqrt(ns_slice_dat_U**2 + ns_slice_dat_V**2)
        ns_slice_x =  nav_lat[:,ii]
        ns_slice_y =  rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,:,ii]
        return ns_slice_dat,ns_slice_x, ns_slice_y

    def reload_hov_data_derived_var_baroc_mag():                
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        if secdataset_proc == 'Dataset 1':
            hov_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][:,:,jj,ii].load()).T
        elif secdataset_proc == 'Dataset 2':
            hov_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,::thin,::thin][:,:,jj,ii].load()).T
        elif secdataset_proc =='Dat2-Dat1':
            hov_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,::thin,::thin][:,:,jj,ii].load()).T
            hov_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,::thin,::thin][:,:,jj,ii].load()).T
        hov_dat = np.sqrt(hov_dat_U**2 + hov_dat_V**2)
        hov_x = time_datetime
        hov_y = rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin][0,:,jj,ii]
        return hov_dat,hov_x,hov_y

    def reload_map_data_derived_var():
        if var == 'baroc_mag':
            if z_meth == 'z_slice':
                map_dat = reload_map_data_derived_var_baroc_mag_zmeth_z_slice()
            elif z_meth in ['ss','nb','df']:
                map_dat = reload_map_data_derived_var_baroc_mag_zmeth_ss_nb_df()
            elif z_meth == 'z_index':
                map_dat = reload_map_data_derived_var_baroc_mag_z_index()
            else:
                print('z_meth not supported:',z_meth)
                pdb.set_trace()
        elif var == 'pea': 
            map_dat = reload_map_data_derived_var_pea()
            '''
            if z_meth == 'z_slice':
                map_dat = reload_map_data_derived_var_pea()
            elif z_meth in ['ss','nb','df']:
                map_dat = reload_map_data_derived_var_pea_zmeth_ss_nb_df()
            elif z_meth == 'z_index':
                map_dat = reload_map_data_derived_var_pea_z_index()
            else:
                print('z_meth not supported:',z_meth)
                pdb.set_trace()
            '''
        else:
            print('var not in deriv_var',var)
        map_x = nav_lon
        map_y = nav_lat
        
        return map_dat,map_x,map_y


    def reload_map_data_derived_var_baroc_mag_zmeth_z_slice():
        if var_dim[var] == 4:
            if secdataset_proc == 'Dataset 1':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
                map_dat_3d_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            map_dat_3d = np.sqrt(map_dat_3d_U**2 + map_dat_3d_V**2)

            if zz not in interp1d_wgtT.keys(): interp1d_wgtT[zz] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,::thin,::thin],zz)
            map_dat =  interp1dmat_wgt(map_dat_3d,interp1d_wgtT[zz])
        
        elif var_dim[var] == 3:
            if secdataset_proc == 'Dataset 1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        return map_dat

    def reload_map_data_derived_var_baroc_mag_zmeth_ss_nb_df():


        if var_dim[var] == 4:
            if secdataset_proc == 'Dataset 1':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
                map_dat_3d_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
            map_dat_3d = np.sqrt(map_dat_3d_U**2 + map_dat_3d_V**2)
            map_dat_ss = map_dat_3d[0,:,:]
            map_dat_nb = np.ma.array(extract_nb(map_dat_3d[:,:,:],nbind),mask = tmask[0,:,:])
            if z_meth == 'ss': map_dat = map_dat_ss
            if z_meth == 'nb': map_dat = map_dat_nb
            if z_meth == 'df': map_dat = map_dat_ss - map_dat_nb
        elif var_dim[var] == 3:
            if secdataset_proc == 'Dataset 1':
            #map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        return map_dat

    def reload_map_data_derived_var_baroc_mag_z_index():
        if var_dim[var] == 4:
            if secdataset_proc == 'Dataset 1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,zz,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,zz,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,zz,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,zz,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,zz,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,zz,::thin,::thin].load())
                map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,zz,::thin,::thin].load())
                map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,zz,::thin,::thin].load())
            map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        elif var_dim[var] == 3:
            if secdataset_proc == 'Dataset 1':
            #map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc == 'Dataset 2':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            elif secdataset_proc =='Dat2-Dat1':
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
            map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        return map_dat




    def reload_map_data_derived_var_pea():

        gdept_mat = rootgrp_gdept.variables['gdept_0'][:,:,::thin,::thin]
        dz_mat = rootgrp_gdept.variables['e3t_0'][:,:,::thin,::thin]

        if secdataset_proc == 'Dataset 1':
            tmp_T_data = np.ma.masked_invalid(curr_tmp_data.variables['votemper'][ti,:,::thin,::thin].load())
            tmp_S_data = np.ma.masked_invalid(curr_tmp_data.variables['vosaline'][ti,:,::thin,::thin].load())
            map_dat = pea_TS(tmp_T_data[np.newaxis],tmp_S_data[np.newaxis],gdept_mat,dz_mat,tmask=tmask[:,::thin,::thin][np.newaxis]==False,calc_TS_comp = False )[0] # tmppea,tmppeat,tmppeas, calc_TS_comp = True
        elif secdataset_proc == 'Dataset 2':
            tmp_T_data = np.ma.masked_invalid(curr_tmp_data_diff.variables['votemper'][ti,:,::thin,::thin].load())
            tmp_S_data = np.ma.masked_invalid(curr_tmp_data_diff.variables['vosaline'][ti,:,::thin,::thin].load())
            map_dat = pea_TS(tmp_T_data[np.newaxis],tmp_S_data[np.newaxis],gdept_mat,dz_mat,tmask=tmask[:,::thin,::thin][np.newaxis]==False,calc_TS_comp = False )[0] # tmppea,tmppeat,tmppeas, calc_TS_comp = True
        elif secdataset_proc =='Dat2-Dat1':
            tmp_T_data = np.ma.masked_invalid(curr_tmp_data.variables['votemper'][ti,:,::thin,::thin].load())
            tmp_S_data = np.ma.masked_invalid(curr_tmp_data.variables['vosaline'][ti,:,::thin,::thin].load())
            map_dat = pea_TS(tmp_T_data[np.newaxis],tmp_S_data[np.newaxis],gdept_mat,dz_mat,tmask=tmask[:,::thin,::thin][np.newaxis]==False,calc_TS_comp = False )[0] # tmppea,tmppeat,tmppeas, calc_TS_comp = True
            tmp_T_data_diff = np.ma.masked_invalid(curr_tmp_data_diff.variables['votemper'][ti,::thin,::thin].load())
            tmp_S_data_diff = np.ma.masked_invalid(curr_tmp_data_diff.variables['vosaline'][ti,::thin,::thin].load())
            map_dat -= pea_TS(tmp_T_data_diff[np.newaxis],tmp_S_data_diff[np.newaxis],gdept_mat,dz_mat,tmask=tmask[:,::thin,::thin][np.newaxis]==False,calc_TS_comp = False )[0] # tmppea,tmppeat,tmppeas, calc_TS_comp = True
        return map_dat
  
    def reload_map_data():
        #pdb.set_trace()
        if var_dim[var] == 3:
            map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
            if load_2nd_files: map_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,::thin,::thin].load())
        else:
            if z_meth == 'z_slice':
                map_dat = reload_map_data_zmeth_zslice()
            elif z_meth in ['ss','nb','df']:
                map_dat = reload_map_data_zmeth_ss_nb_df()
            elif z_meth == 'z_index':
                map_dat = reload_map_data_zmeth_zindex()
            else:
                print('z_meth not supported:',z_meth)
                pdb.set_trace()

        map_x = nav_lon
        map_y = nav_lat
        
        return map_dat,map_x,map_y
                



    def reload_map_data_zmeth_zslice():
        if secdataset_proc == 'Dataset 1':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
        elif secdataset_proc == 'Dataset 2':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())
        elif secdataset_proc =='Dat2-Dat1':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
            map_dat_3d -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())

        if zz not in interp1d_wgtT.keys(): interp1d_wgtT[zz] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,::thin,::thin],zz)
        map_dat =  interp1dmat_wgt(map_dat_3d,interp1d_wgtT[zz])
        
        return map_dat


            
    def reload_map_data_zmeth_ss_nb_df():
        if secdataset_proc == 'Dataset 1':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
        elif secdataset_proc == 'Dataset 2':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())
        elif secdataset_proc =='Dat2-Dat1':
            map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
            map_dat_3d -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())
        map_dat_ss = map_dat_3d[0]
        map_dat_nb = np.ma.array(extract_nb(map_dat_3d,nbind[:,::thin,::thin]),mask = tmask[0,::thin,::thin])
        if z_meth == 'ss': map_dat = map_dat_ss
        if z_meth == 'nb': map_dat = map_dat_nb
        if z_meth == 'df': map_dat = map_dat_ss - map_dat_nb
        return map_dat

    def reload_map_data_zmeth_zindex():
        if secdataset_proc == 'Dataset 1':
            map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,zz,::thin,::thin].load())
        elif secdataset_proc == 'Dataset 2':
            map_dat = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,zz,::thin,::thin].load())
        elif secdataset_proc =='Dat2-Dat1':
            map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,zz,::thin,::thin].load())
            map_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,zz,::thin,::thin].load())
        return map_dat

    
    ###########################################################################
    # Inner functions defined
    ###########################################################################
    
    if verbose_debugging: print('Inner functions created ', datetime.now())

    #pdb.set_trace()
    #get the current xlim (default to None??)
    cur_xlim = xlim
    cur_ylim = ylim
    # only load data when needed
    reload_map, reload_ew, reload_ns, reload_hov, reload_ts = True,True,True,True,True


    if verbose_debugging: print('Create interpolation weights ', datetime.now())
    if z_meth_default == 'z_slice':
        interp1d_wgtT = {}
        interp1d_wgtT[0] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,::thin,::thin],0)
    if verbose_debugging: print('Interpolation weights created', datetime.now())
    # loop


    if verbose_debugging: print('Start While Loop', datetime.now())
    if verbose_debugging: print('')
    if verbose_debugging: print('')
    if verbose_debugging: print('')

    # initialise button press location
    tmp_press = [(0.5,0.5,)]

    while ii is not None:
        # try, exit on error
        #try:
        if True: 
            # extract plotting data (when needed), and subtract off difference files if necessary.

            
            if verbose_debugging: print('Set current data set (set of nc files) for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())

            if var_grid[var] == 'T':
                curr_tmp_data = tmp_data
                if load_2nd_files: curr_tmp_data_diff = tmp_data_diff
            elif var_grid[var] == 'U':
                curr_tmp_data = tmp_data_U
                if load_2nd_files: curr_tmp_data_diff = tmp_data_diff_U
            elif var_grid[var] == 'V':
                curr_tmp_data = tmp_data_V
                if load_2nd_files: curr_tmp_data_diff = tmp_data_diff_V
            elif var_grid[var] == 'UV':
                curr_tmp_data_U = tmp_data_U
                curr_tmp_data_V = tmp_data_V
                if load_2nd_files: curr_tmp_data_diff_U = tmp_data_diff_U
                if load_2nd_files: curr_tmp_data_diff_V = tmp_data_diff_V
            else:
                print('grid dict error')
                pdb.set_trace()

                
            if verbose_debugging: print('Reload data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())

            if reload_map:
                if var in deriv_var:
                    map_dat,map_x,map_y = reload_map_data_derived_var()
                else:
                    map_dat,map_x,map_y = reload_map_data()
                reload_map = False
            if verbose_debugging: print('Reloaded map data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if reload_ew:
                if var_dim[var] == 4:
                    
                    if var in deriv_var:
                        ew_slice_dat,ew_slice_x, ew_slice_y = reload_ew_data_derived_var()
                    else:
                        ew_slice_dat,ew_slice_x, ew_slice_y = reload_ew_data()

                reload_ew = False
            if verbose_debugging: print('Reloaded ew data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if reload_ns:
                if var_dim[var] == 4:
                    if var in deriv_var:
                        ns_slice_dat,ns_slice_x, ns_slice_y = reload_ns_data_derived_var()    
                    else:
                        ns_slice_dat,ns_slice_x, ns_slice_y = reload_ns_data()                    
                reload_ns = False
            if verbose_debugging: print('Reloaded ns data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if reload_hov:
                if var_dim[var] == 4:
                    if var in deriv_var:
                        hov_dat,hov_x,hov_y = reload_hov_data_derived_var()
                    else:
                        hov_dat,hov_x,hov_y = reload_hov_data()


                reload_hov = False
            if verbose_debugging: print('Reloaded hov data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if reload_ts:
                #if var_grid[var] != 'UV':
                
                if var in deriv_var:
                    ts_x = np.ma.ones(len(nctime))*np.ma.masked
                elif var not in deriv_var:
                    if var_dim[var] == 4:
                    
                        ts_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,zi,jj,ii].load())
                        if load_2nd_files:
                            ts_dat_1 = np.ma.masked_invalid(curr_tmp_data.variables[var][:,zi,jj,ii].load())
                            ts_dat_2 = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,zi,jj,ii].load())
                    elif var_dim[var] == 3:
                        ts_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,jj,ii].load())
                        if load_2nd_files:
                            ts_dat_1 = np.ma.masked_invalid(curr_tmp_data.variables[var][:,jj,ii].load())
                            ts_dat_2 = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,jj,ii].load())
                    ts_x = time_datetime

                    if z_meth in ['ss','nb','df']:
                        ss_ts_dat = curr_tmp_data.variables[var][:,0,jj,ii].load()
                        nb_ts_dat = curr_tmp_data.variables[var][:,np.where(nbind[:,jj,ii] == False)[0],jj,ii].load()
                        df_ts_dat = ss_ts_dat - nb_ts_dat

                        if z_meth == 'ss':ts_dat = ss_ts_dat
                        if z_meth == 'nb':ts_dat = nb_ts_dat
                        if z_meth == 'df':ts_dat = df_ts_dat
                        if load_2nd_files:
                            ss_ts_dat_2 = curr_tmp_data_diff.variables[var][:,0,jj,ii].load()
                            nb_ts_dat_2 = curr_tmp_data_diff.variables[var][:,np.where(nbind[:,jj,ii] == False)[0],jj,ii].load()
                            df_ts_dat_2 = ss_ts_dat_2 - nb_ts_dat_2

                            if z_meth == 'ss':ts_dat_1 = ss_ts_dat
                            if z_meth == 'nb':ts_dat_1 = nb_ts_dat
                            if z_meth == 'df':ts_dat_1 = df_ts_dat
                            if z_meth == 'ss':ts_dat_2 = ss_ts_dat_2
                            if z_meth == 'nb':ts_dat_2 = nb_ts_dat_2
                            if z_meth == 'df':ts_dat_2 = df_ts_dat_2
                reload_ts = False
                
            if verbose_debugging: print('Reloaded ts data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
                
                
            if verbose_debugging: print('Reloaded data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())


            
            if verbose_debugging: print('Choose cmap based on secdataset_proc:',secdataset_proc, datetime.now())

            # Choose the colormap depending on which dataset being shown
            if secdataset_proc == 'Dat2-Dat1':
                curr_cmap = diff_cmap
                clim_sym = True
            elif secdataset_proc in ['Dataset 1','Dataset 2']:
                curr_cmap = base_cmap
                clim_sym = False

            #plot data
            pax = []
            #pdb.set_trace()
        
            
            if verbose_debugging: print("Do pcolormesh for ii = %i,jj = %i,ti = %i,zz = %i, var = '%s'"%(ii,jj, ti, zz,var), datetime.now())
            pax.append(ax[0].pcolormesh(map_x,map_y,map_dat,cmap = curr_cmap,norm = climnorm))
            if var_dim[var] == 4:
                pax.append(ax[1].pcolormesh(ew_slice_x,ew_slice_y,ew_slice_dat,cmap = curr_cmap,norm = climnorm))
                pax.append(ax[2].pcolormesh(ns_slice_x,ns_slice_y,ns_slice_dat,cmap = curr_cmap,norm = climnorm))
                pax.append(ax[3].pcolormesh(hov_x,hov_y,hov_dat,cmap = curr_cmap,norm = climnorm))
            tsax = ax[4].plot(ts_x,ts_dat,'r')
            # add variable name as title - maybe better as a button color chnage?
            ax[0].set_title('%s (%i, %i, %i, %i) '%(var,ii,jj,zz,ti))
            
            if verbose_debugging: print('Set limits ', datetime.now())
            # add colorbars
            if verbose_debugging: print('add colorbars', datetime.now(), 'len(ax):',len(ax))            
            cax = []      
            if var_dim[var] == 4:  
                for ai in [0,1,2,3]: cax.append(plt.colorbar(pax[ai], ax = ax[ai]))
            elif var_dim[var] == 3:
                for ai in [0]: cax.append(plt.colorbar(pax[ai], ax = ax[ai]))
            if verbose_debugging: print('added colorbars', datetime.now(), 'len(ax):',len(ax),'len(cax):',len(cax))
            
            # apply xlim/ylim if keyword set
            if cur_xlim is not None:ax[0].set_xlim(cur_xlim)
            if cur_ylim is not None:ax[0].set_ylim(cur_ylim)
            if cur_xlim is not None:ax[1].set_xlim(cur_xlim)
            if cur_ylim is not None:ax[2].set_xlim(cur_ylim)
            if tlim is not None:ax[3].set_xlim(tlim)
            if tlim is not None:ax[4].set_xlim(tlim)
            
            #reset ylim to time series to data min max
            ax[4].set_ylim(ts_dat.min(),ts_dat.max())



            # set minimum depth if keyword set
            xlim_min = 1
            if zlim_max == None:
                #pdb.set_trace()
                tmpew_xlim = ax[1].get_xlim()
                tmpns_xlim = ax[2].get_xlim()
                tmpew_visible_ind = (ew_slice_x>=tmpew_xlim[0]) & (ew_slice_x<=tmpew_xlim[1]) 
                tmpns_visible_ind = (ns_slice_x>=tmpns_xlim[0]) & (ns_slice_x<=tmpns_xlim[1]) 


                ax[1].set_ylim([ew_slice_y[:,tmpew_visible_ind].max(),xlim_min])
                ax[2].set_ylim([ns_slice_y[:,tmpns_visible_ind].max(),xlim_min])
                ax[3].set_ylim([hov_y.max(),xlim_min])
            else:
                ax[1].set_ylim([zlim_max,xlim_min])
                ax[2].set_ylim([zlim_max,xlim_min])
                ax[3].set_ylim([np.minimum(zlim_max,hov_y.max()),xlim_min])


            #print('About to reset colour limits')
            # if no keyword clim, use 5th and 95th percentile of data        
            #for tmpax in ax[:-1]:print('current clim',get_clim_pcolor(ax = tmpax))    
            #pdb.set_trace()
            try:
                if clim is None:
                    for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)
                    #When using the log scale, the colour set_clim seems linked, so all panels get set to the limits of the final set_perc_clim_pcolor call..
                    #   therefore repeat set_perc_clim_pcolor of the map, so the hovmuller colour limit is not the final one. 
                    set_perc_clim_pcolor_in_region(5,95, ax = ax[0],sym = clim_sym)

                else:
                    for tmpax in ax[:-1]:set_clim_pcolor((clim), ax = tmpax)
            except:
                print("An exception occured - probably 'IndexError: cannot do a non-empty take from an empty axes.'")
                pdb.set_trace()
        
            #for tmpax in ax[:-1]:print('updated clim',get_clim_pcolor(ax = tmpax))    
        
            #print('Have reset colour limits')


            if verbose_debugging: print('Plot location lines for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            
            ## add lines to show current point. 
            # using plot for the map to show lines if on a rotated grid (amm15) etc.
            cs_plot_1 = ax[0].plot(nav_lon[jj,:],nav_lat[jj,:],color = '0.5', alpha = 0.5) 
            cs_plot_2 = ax[0].plot(nav_lon[:,ii],nav_lat[:,ii],color = '0.5', alpha = 0.5)
            cs_line = []
            # using axhline, axvline, for slices, hov, time series
            #cs_line.append(ax[0].axhline(nav_lat[jj,ii],color = '0.5', alpha = 0.5))
            #cs_line.append(ax[0].axvline(nav_lon[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[1].axvline(nav_lon[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[2].axvline(nav_lat[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[3].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))
            cs_line.append(ax[4].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))
            cs_line.append(ax[1].axhline(zz,color = '0.5', alpha = 0.5))
            cs_line.append(ax[2].axhline(zz,color = '0.5', alpha = 0.5))
            cs_line.append(ax[3].axhline(zz,color = '0.5', alpha = 0.5))

            # Redraw canvas
            #==================
            fig.canvas.draw()
            fig.canvas.flush_events()

            # set current axes to hidden full screen axes for click interpretation
            plt.sca(clickax)
            

            
            #await click with ginput
            if verbose_debugging: print('Waiting for button press', datetime.now())
            if verbose_debugging: print('mode', mode,'mouse_in_Click',mouse_in_Click,datetime.now())
            

            #if mouse_in_Click:
            #    pdb.set_trace()

            if mode == 'Loop':
                if mouse_in_Click:
                    mode = 'Click'
                    but_name = 'Click'
                    func_but_text_han['Click'].set_color('gold')
                    func_but_text_han['Loop'].set_color('k')

            if mode == 'Click':
                tmp_press = plt.ginput(1)
            # if tmp_press is empty (button press detected from another window, persist previous location. 
            #    Previously a empty array led to a continue, which led to the bug where additional colorbar were added
            if len(tmp_press) == 0:
                press_ginput = press_ginput
            else:
                press_ginput = tmp_press
            if verbose_debugging: print('')
            if verbose_debugging: print('')
            if verbose_debugging: print('')
            if verbose_debugging: print('Button pressed!', datetime.now())

            clii,cljj = press_ginput[0][0],press_ginput[0][1]

            if verbose_debugging: print("selected clii = %f,cljj = %f"%(clii,cljj))

            #get click location, and current axis limits for ax[0], and set them
            # defunct? was trying to allow zooming
            cur_xlim = np.array(ax[0].get_xlim())
            cur_ylim = np.array(ax[0].get_ylim())

            ax[0].set_xlim(cur_xlim)
            ax[0].set_ylim(cur_ylim)

            #find clicked axes:
            is_in_axes = False
            
            # convert the mouse click into data indices, and report which axes was clicked
            sel_ax,sel_ii,sel_jj,sel_ti,sel_zz = indices_from_ginput_ax(clii,cljj, thin = thin)

            
                
            #pdb.set_trace()
            if verbose_debugging: print("selected sel_ax = %s,sel_ii = %s,sel_jj = %s,sel_ti = %s,sel_zz = %s"%(sel_ax,sel_ii,sel_jj,sel_ti,sel_zz))

            #print(sel_ax,sel_ii,sel_jj,sel_ti,sel_zz )

            if sel_ax is not None :  is_in_axes = True 

            if sel_ax == 0:               
                ii = sel_ii
                jj = sel_jj

                # and reload slices, and hovmuller/time series
                reload_ew = True
                reload_ns = True
                reload_hov = True
                reload_ts = True

            elif sel_ax in [1]: 
                ii = sel_ii
                # if in ew slice, change ns slice, and hov/time series
                
                reload_ns = True
                reload_hov = True
                reload_ts = True
                
            elif sel_ax in [2]:
                jj = sel_jj
                # if in ns slice, change ew slice, and hov/time series

                reload_ew = True
                reload_hov = True
                reload_ts = True

            elif sel_ax in [3]:
                # if in hov/time series, change map, and slices

                # re calculate depth values, as y scale reversed, 
                zz = sel_zz
                z_meth = z_meth_default

                
                reload_map = True
                reload_ts = True

            elif sel_ax in [4]:
                # if in hov/time series, change map, and slices
                ti = sel_ti
                
                reload_map = True
                reload_ew = True
                reload_ns = True
   
            
            if mode == 'Loop':
                ti+=1
                if ti == ntime: 
                    ti = 0
                    #mode = 'Click'
                    #pdb.set_trace()
                    

                    
                
            
            if verbose_debugging: print('Decide what to reload', datetime.now())

            if verbose_debugging: print("selected ii = %s,jj = %s,ti = %s,zz = %s"%(ii,jj,ti,zz))

            # if in button, change variables. 
            
            for but_name in but_extent.keys():
                
                but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = but_extent[but_name]
                if (clii >= but_pos_x0) & (clii <= but_pos_x1) & (cljj >= but_pos_y0) & (cljj <= but_pos_y1):
                    is_in_axes = True
                    if but_name in var_mat:
                        var = but_name


                        if var_dim[var] == 3:
                            z_meth = z_meth_default

                            func_but_text_han['Depth level'].set_color('r')
                            func_but_text_han['Surface'].set_color('k')
                            func_but_text_han['Near-Bed'].set_color('k')
                            func_but_text_han['Surface-Bed'].set_color('k')
                        
                        for vi,var_dat in enumerate(var_mat): but_text_han[var_dat].set_color('k')
                        but_text_han[but_name].set_color('r')
                        fig.canvas.draw()
                        
                        climnorm = None 
                        func_but_text_han['Clim: log'].set_color('k')
                        func_but_text_han['Clim: normal'].set_color('b')

                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True

            for but_name in func_but_extent.keys():
                
                but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = func_but_extent[but_name]
                if (clii >= but_pos_x0) & (clii <= but_pos_x1) & (cljj >= but_pos_y0) & (cljj <= but_pos_y1):
                    is_in_axes = True
                    print(but_name)
                    if but_name in 'Reset zoom':
                        # set xlim and ylim to max size possible from nav_lat and nav_lon
                        cur_xlim = np.array([nav_lon.min(),nav_lon.max()])
                        cur_ylim = np.array([nav_lat.min(),nav_lat.max()])
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True
                    elif but_name in 'Zoom':
                        # use ginput to take two clicks as zoom region. 
                        # only coded for main axes
                        
                        
                        plt.sca(clickax)
                        tmpzoom = plt.ginput(2)

                        # sort the zoom clicks, so that the x and y lims are the right way around. 
                        tmpzoom_array = tmp = np.array(tmpzoom)     
                        tmpzoom_array.sort(axis = 0)
                        tmpzoom_sorted = [tuple(tmp[0,:]), tuple(tmp[1,:])]
                        tmpzoom = tmpzoom_sorted
                        #pdb.set_trace()
                        
                        #convert clicks to data indices
                        zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz = indices_from_ginput_ax(tmpzoom[0][0],tmpzoom[0][1], thin = thin)
                        zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz = indices_from_ginput_ax(tmpzoom[1][0],tmpzoom[1][1], thin = thin)
                        if verbose_debugging: print(zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz)
                        if verbose_debugging: print(zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz)
                        if verbose_debugging: print(cur_xlim)
                        if verbose_debugging: print(cur_ylim)
                        # if both clicks in main axes, use clicks for the new x and ylims
                        if (zoom0_ax is not None) & (zoom0_ax is not None):
                            if zoom0_ax == zoom1_ax:
                                if zoom0_ax == 0:
                                    cur_xlim = np.array([nav_lon[zoom0_jj,zoom0_ii],nav_lon[zoom1_jj,zoom1_ii]])
                                    cur_ylim = np.array([nav_lat[zoom0_jj,zoom0_ii],nav_lat[zoom1_jj,zoom1_ii]])
                                    reload_map = True
                                    reload_ew = True
                                    reload_ns = True
                                    reload_hov = True
                                    reload_ts = True
                                
                        #print(cur_xlim)
                        #print(cur_ylim)

                    
                    elif but_name == 'Clim: Reset':
                        clim = None
                    
                    elif but_name == 'Clim: Zoom': 


                        plt.sca(clickax)
            
                        tmpczoom = plt.ginput(2)
                        clim = np.array([tmpczoom[0][1],tmpczoom[1][1]])
                        clim.sort()


                    elif but_name == 'Clim: Expand': 
                        clim = np.array(get_clim_pcolor(ax = ax[0]))
                        if climnorm is None:
                            clim = np.array([clim.mean() - clim.ptp(),clim.mean() + clim.ptp()])
                        else:
                            clim = np.log10(np.array([(10**clim).mean() - (10**clim).ptp(),(10**clim).mean() + (10**clim).ptp()]))
                        
                    
                    elif but_name == 'Clim: perc': 
                        clim = None

                    
                    elif but_name == 'Clim: log': 
                        if secdataset_proc == 'Dat2-Dat1':
                            func_but_text_han['Clim: log'].set_color('0.5')
                        else:
                            #clim = get_clim_pcolor(ax = ax[0])
                            #pdb.set_trace()
                            #climnorm = matplotlib.colors.LogNorm(tmpclim[0],tmpclim[1]) # matplotlib.colors.LogNorm(0.005,0.1)
                            climnorm = matplotlib.colors.LogNorm() # matplotlib.colors.LogNorm(0.005,0.1)
                            func_but_text_han['Clim: log'].set_color('k')
                            func_but_text_han['Clim: normal'].set_color('k')
                            func_but_text_han[but_name].set_color('b')
                        

                    elif but_name == 'Clim: normal':
                        climnorm = None 
                        func_but_text_han['Clim: log'].set_color('k')
                        func_but_text_han['Clim: normal'].set_color('k')
                        func_but_text_han[but_name].set_color('b')


                    elif but_name in secdataset_proc_list:
                        secdataset_proc = but_name
                        func_but_text_han['Dat2-Dat1'].set_color('k')
                        func_but_text_han['Dataset 1'].set_color('k')
                        func_but_text_han['Dataset 2'].set_color('k')
                        func_but_text_han[but_name].set_color('darkgreen')
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True

                        #if changing to a difference plot, change to clim normal
                        if but_name == 'Dat2-Dat1':
                            func_but_text_han['Clim: log'].set_color('0.5')   
                            climnorm = None 
                            func_but_text_han['Clim: normal'].set_color('b')
                        else:
                            func_but_text_han['Clim: log'].set_color('k')   
                    




                    elif but_name in ['Surface','Near-Bed','Surface-Bed']:
                        if var_dim[var] == 4:
                            
                            '''
                            zz = 0
                            map_dat_3d = np.ma.masked_invalid(tmp_data.variables[var][ti,:,::thin,::thin].load())
                            map_dat_ss = map_dat_3d[0,:,:]
                            map_dat_nb = np.ma.array(extract_nb(map_dat_3d[:,:,:],nbind),mask = tmask[0,:,:])
                            if but_name == 'Surface': map_dat = map_dat_ss
                            if but_name == 'Near-Bed': map_dat = map_dat_nb
                            if but_name == 'Surface-Bed': map_dat = map_dat_ss - map_dat_nb

                            '''
                            
                            if but_name == 'Surface':z_meth = 'ss'
                            if but_name == 'Near-Bed': z_meth = 'nb'
                            if but_name == 'Surface-Bed': z_meth = 'df'
                            reload_map = True
                            reload_ts = True


                            func_but_text_han['Depth level'].set_color('k')
                            func_but_text_han['Surface'].set_color('k')
                            func_but_text_han['Near-Bed'].set_color('k')
                            func_but_text_han['Surface-Bed'].set_color('k')
                            func_but_text_han[but_name].set_color('r')
                            fig.canvas.draw()

                            #pdb.set_trace()
                    elif but_name in ['Depth level']:
                        func_but_text_han['Depth level'].set_color('k')
                        func_but_text_han['Surface'].set_color('k')
                        func_but_text_han['Near-Bed'].set_color('k')
                        func_but_text_han['Surface-Bed'].set_color('k')
                        func_but_text_han[but_name].set_color('r')
                        z_meth = z_meth_default    
                        reload_map = True
                        reload_ts = True
                    elif but_name in 'Save Figure':                        
                        #fig.savefig('/home/h01/hadjt/workspace/python3/tmpfig_%i%i_%i_%i_%s.png'%(ii,jj,ti,zz))
                        if not os.path.exists(fig_dir):
                            os.makedirs(directory)

                        if fig_cutout:
                            bbox_inches =  matplotlib.transforms.Bbox([[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.05-0.01)],[fig.get_figwidth()*(func_but_x0-0.01),fig.get_figheight()*0.95]])
                            fig.savefig('%s/output_%s_%s_%i_%i_%i_%i_%s.png'%(fig_dir,fig_lab,var,ii,jj,ti,zz,z_meth),bbox_inches = bbox_inches)
                        else:
                            fig.savefig('%s/output_%s_%s_%i_%i_%i_%i_%s.png'%(fig_dir,fig_lab,var,ii,jj,ti,zz,z_meth))



                    elif but_name in mode_name_lst:
                        '''
                        # Careful Catch to change mode to Click at the end of the loop, to avoid and infinte loop
                        if (ti == 0) & (mode == 'Loop'): 
                            mode = 'Click'
                        else:
                            mode = but_name
                        '''
                        if mode == 'Loop': 
                            #pdb.set_trace()
                            mouse_in_Click = False
                        mode = but_name
                        func_but_text_han['Click'].set_color('k')
                        func_but_text_han['Loop'].set_color('k')
                        func_but_text_han[mode].set_color('gold')
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True
                    elif but_name in 'Quit':
                        print('Quit')
                        print('')
                        print('')
                        print('')
                        return
                    else:
                        print(but_name)
                        print('No function for but_name')
                        pdb.set_trace()
                    print(clim)
                        
                        

            plt.sca(ax[0])
                    


            print("selected ii = %i,jj = %i,ti = %i,zz = %i, var = '%s'"%(ii,jj, ti, zz,var))
            # after selected indices and vareiabels, delete plots, ready for next cycle
            #pdb.set_trace()
            for tmp_cax in cax:tmp_cax.remove()


            for tmp_pax in pax:tmp_pax.remove()
            for tmp_cs_line in cs_line:tmp_cs_line.remove()
            #for tmp_cs_plot in cs_plot:tmp_cs_plot.remove()
            rem_loc = tsax.pop(0)
            rem_loc.remove()
            cs_plot_1_pop = cs_plot_1.pop()
            cs_plot_1_pop.remove()
            cs_plot_2_pop = cs_plot_2.pop()
            cs_plot_2_pop.remove()
            
            # sometime when it crashes, it adds additional colorbars. WE can catch this be removing any colorbars from the figure... 
            #   however, this doesn't reset the axes size, so when the new colorbar is added, the axes is reduced in size. 
            #   maybe better to specify axes and colobar location, rathar than using subplot, and colorbar().
            for child in fig.get_children():
                child.__class__.__name__
                if child.get_label() == '<colorbar>': child.remove()




def main():
    

    nemo_slice_zlev_helptext=textwrap.dedent('''\
    Interactive NEMO ncfile viewer.
    ===============================
    Developed by Jonathan Tinker Met Office, UK, December 2023
    ==========================================================
    
    When calling from the command line, it uses a mix of positional values, and keyword value pairs, via argparse.

    The first two positional keywords are the NEMO configuration "config", 
    and the second is the list of input file names "fname_lst"
    
    config: should be AMM7, AMM15, ORCA025, ORCA025EXT or ORCA12. Other configurations will be supported soon. 
    fname_lst: supports wild cards, but should be  enclosed in quotes.

    e.g.
    python NEMO_nc_slevel_viewer_dev.py amm15 "/scratch/frpk/a15ps46trial/control/prodm_op_am-dm.gridT*-36.nc" 


    Optional arguments are give as keyword value pairs, with the keyword following a double hypen.
    We will list the most useful options first.

    --zlim_max - maximum depth to show, often set to 200. Default is None
    
    --subtracted_flist - secondary file list, to show the different between two sets of files. 
        Enclose in quotes. Make sure this has the same number of files, with the same dates as 
        fname_lst. This will be checked in later upgrades, but will currently fail if the files
        are inconsistent
    --U_flist - specify a consistent set of U and V files, to calculate a drived variable current magintude. 
        assumes the variable vozocrtx is present. Later upgrade will allow the plotting of vectors, 
        and to handle other current variable names. Must have both U_flist and V_flist.
    --V_flist - specify a consistent set of U and V files, to calculate a drived variable current magintude. 
        assumes the variable vomecrty is present. Later upgrade will allow the plotting of vectors, 
        and to handle other current variable names. Must have both U_flist and V_flist.


    --fig_dir - directory for figure output
    --fig_lab - label to add to filesnames, so can compare runs.
    --fig_cutout - save full screen, or cut off the buttons - this is the defaulted to True

    --clim_sym use a symetrical colourbar -defaulted to False
    --use_cmocean - use cmocean colormaps -defaulted to False

    --verbose_debugging - prints out lots of statements at run time, to help debug -defaulted to False

    --ii    initial ii value
    --jj    initial jj value
    --ti    initial ti value
    --zz    initial zz value


    --thin  thin the data, to only load the xth row and column

    Planned upgrades:
    =================
    Button to switch between flist, subtracted_flist and their difference.
    Plot current vectors.
    Improve meaningfulness of the figure title. State level being plotted (zlev, ss, df etc.)
    Allow colorbar to be specified
    Work on CRAY

        Allow the keyword thin, to think the data before plotting, to speed up large datasets like amm15
        Additional derived variables (PEA)


    Using NEMO_nc_slevel_viewer.
    ============================

    BUG
    ===
    BUG: sometimes additional colorbars start to appear. Clicking somewhere tends to remove them, although sometimes you get additional cross-hairs.
        it maybe easiest to quit and start again. 

    
    Overview
    ========
    When the viewer loads, there is a series of variable buttons on the left hand side, fuction buttons on the right hand side, and subplots.
    The main subplot is on the left, which is  2d lon lat surface plot. The right hand plots (top to bottom) show a zonal and meridonial slice, 
    a hovmoller plot and a timeseries. The active location is showns as crosshairs on each subplot. 

    Changing the current location
    =============================
    You can change the horizontal location by clicking on the map, or the either of the slices. YOu can change the depth by clicking on the hovmuller plots, 
    and change the time by clicking on the time series. 

    The viewer initially shows the surface slice, and the viewer is in depth level mode (Note the Depth Level button is red on the right hand side). 
    When you change the depth by clicking on the hovmuller diagram, you take a z slice though AMM scoorinates, therefore the coastline 
    changes when you go deeper. If you want to see the surface, the bed, or the surface minus bed, you can click the buttons on the right hand side.
    these will change to red depending on your choice - remembering to click twice.  When you wnat to go back to the depth level, click back on depth level.
    
    
    Changing variables
    ==================
    You can change variables by clicking (twice) on the variable buttons on the left hand side. the current variable is in red. 3d variables are have black boxes
    2d variables have green boxes, derived variables have blue boxes
    
    Changing Datasets
    ==================
    You can load two data sets using --subtracted_flist, and then switch between the dataset, and show there differnce with the "Dataset 1", "Dataset 2", "Dat2-Dat1" buttons.

    Loop and Click Modes
    ==================
    The default mode is "Click", to use the mouse to click the buttons, and figures... in this mode, the program awaits a mouse click to continue. 
    The other mode is "Loop", where the timeslices are looped throught. This mode does not await a mouse click to continue, 
    and so clickng on the other buttons will not exit this mode. Instead, we track the mouse location, and see when you point to the "Click" button, and wait.
    The next iteration will allow you to click on "Click" to continue. 
    
    Zooming
    =======

    You can zoom, by .
        1) clicking on the zoom button, 
        2) clicking on the map at the bottom left hand point of your area of interest, 
        3) clicking on the map at the top right hand point of your area of interest, 
        4) clicking on some white space.

    You can reset the zoom by clicking Reset zoom, and the white space
    

    Colour Zooming
    ==============
    The default colormap limits are based on the 5th and 95th percentile value that occurs within the subplot.
    If you can want tighter colour limits you can zoom the colorbar, reset to the original default values, or zoom out.
    
    To zoom in on the colourbar
        1)  Click Clim: Zoom
        2)  Click on the colorbar of the map (left hand subplot) at the desired lower colour limit
        3)  Click on the desired upper colour limit of the colorbar.
        4)  Click on white space.

    To zoom out of the colorbar (double the colorbar range, with the same middle value)
        1) Click Clim: Expand
        2) Click whitespace
    
    To reset the default colorbar limits
        1) Click Clim: Reset
    
    Colourmap scaling
    =================
    You can set the colorbar to logarithmic or normal.
    However there appears to be a matplotlib bug with logarithmic colorbars
        All the colorbars share the same colour limits when in log scale. 

    It doesn't appear to work when comparing two sets of files.
    It doens't handle negative values very well. 
    
    Data Thinning
    =============
    To speed up handling of large files, you can "thin" the data, only loading every x row and column of the data:
        data[::thin,::thin]

    use the option --thin 5


    Saving figures
    ==============
    You can take snap shots of the screen by clicking Save Figure, and then clicking white space. 
    Files will be saved in the dirertory given with the --fig_dir option.
    Figures will be named based on the variable, ii,jj, ti and zz location, and with a figure label
    given with the --fig_lab option. By default, the savedfigure will exclude the buttons. If you want
    the full screen (or the cut out is not optimised) use  the --fig_cutout False option.

    Quit
    ====
    Click quit, then white space. The figure should close. 


    Developed by Jonathan Tinker Met Office, UK, December 2023
    ==========================================================
    
    ''')

    if sys.argv.__len__() > 1:

        #https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
        parser = argparse.ArgumentParser(description='An interactive tool for plotting NEMO data',
            formatter_class=argparse.RawDescriptionHelpFormatter,  
            epilog=nemo_slice_zlev_helptext)


        parser.add_argument('config', type=str, help="AMM7, AMM15, ORCA025, ORCA025EXT or ORCA12")# Parse the argument
        parser.add_argument('fname_lst', type=str, help='Input file list, enclose in "" more than simple wild card')
        parser.add_argument('--zlim_max', type=int, required=False)
        parser.add_argument('--subtracted_flist', type=str, required=False, help='Input file list, enclose in "" more than simple wild card, Check this has the same number of files as the fname_lst')
        parser.add_argument('--U_flist', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vozocrtx, enclose in "" more than simple wild card')
        parser.add_argument('--V_flist', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vomecrty, enclose in "" more than simple wild card')
        parser.add_argument('--fig_dir', type=str, required=False, help = 'if absent, will default to /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer/tmpfigs')
        parser.add_argument('--fig_lab', type=str, required=False, help = 'if absent, will default to figs')
        parser.add_argument('--fig_cutout', type=bool, required=False)

        parser.add_argument('--ii', type=int, required=False)
        parser.add_argument('--jj', type=int, required=False)
        parser.add_argument('--ti', type=int, required=False)
        parser.add_argument('--zz', type=int, required=False)
        parser.add_argument('--clim_sym', type=bool, required=False)
        parser.add_argument('--use_cmocean', type=bool, required=False)
        parser.add_argument('--thin', type=int, required=False)
        parser.add_argument('--verbose_debugging', type=bool, required=False)

        

        #thin = 1,
        #xlim = None, ylim = None, tlim = None, clim = None,

        args = parser.parse_args()# Print "Hello" + the user input argument

        if args.fig_dir is None: args.fig_dir='/home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer/tmpfigs'
        if args.fig_lab is None: args.fig_lab='figs'
        if args.fig_cutout is None: args.fig_cutout=True
        if args.verbose_debugging is None: args.verbose_debugging=False
        

        if args.thin is None: args.thin=1
        #Deal with file lists

        fname_lst = glob.glob(args.fname_lst)
        fname_lst.sort()
        subtracted_flist = None
        U_flist = None
        V_flist = None

        if args.subtracted_flist is not None:subtracted_flist = glob.glob(args.subtracted_flist)
        if args.U_flist is not None:U_flist = glob.glob(args.U_flist)
        if args.V_flist is not None:V_flist = glob.glob(args.V_flist)

        if subtracted_flist is not None:subtracted_flist.sort()
        if U_flist is not None:U_flist.sort()
        if V_flist is not None:V_flist.sort()


        nemo_slice_zlev(fname_lst,zlim_max = args.zlim_max, config = args.config,
            subtracted_flist = subtracted_flist, U_flist = U_flist, V_flist = V_flist,
            clim_sym = args.clim_sym, use_cmocean = args.use_cmocean,
            thin = args.thin ,
            ii = args.ii, jj = args.jj, ti = args.ti, zz = args.zz, 
            fig_dir = args.fig_dir, fig_lab = args.fig_lab,fig_cutout = args.fig_cutout,
            verbose_debugging = args.verbose_debugging)


        exit()

if __name__ == "__main__":
    main()


