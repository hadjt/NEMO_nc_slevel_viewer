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


sys.path.append('/net/home/h01/hadjt/workspace/python3/')
#sys.path.append('/home/d05/hadjt/scripts/python/')


from NEMO_nc_slevel_viewer_lib import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region,interp1dmat_wgt, interp1dmat_create_weight, nearbed_index,extract_nb,mask_stats,load_nearbed_index

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



import socket
computername = socket.gethostname()
comp = 'linux'
if computername in ['xcel00','xcfl00']: comp = 'hpc'

if comp == 'linux': sys.path.append('/home/d05/hadjt/scripts/python/')



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
    U_flist = None,V_flist = None):


    if use_cmocean:
        
        import cmocean
        # default color map to use
        curr_cmap = None
        diff_cmap = cmocean.cm.balance
    else:
        curr_cmap = None
        #diff_cmap = matplotlib.cm.seismic
        diff_cmap = matplotlib.cm.coolwarm


    if clim_sym is None: clim_sym = False

    # default initial indices
    if ii is None: ii = 120
    if jj is None: jj = 120
    if ti is None: ti = 0
    if zz is None: zz = 0
    if zz == 0: zi = 0


    load_diff_files = False
    # repeat if comparing two time series. 
    if subtracted_flist is not None:
        
        load_diff_files = True



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


        '''
        nav_lon = np.ma.masked_invalid(rootgrp_gdept.variables['glamt'][0])
        nav_lat = np.ma.masked_invalid(rootgrp_gdept.variables['gphit'][0])
        '''
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
        
        
    else:
        nav_lat = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
        nav_lon = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())
        
 
    '''
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
    '''
    
    
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

            if load_diff_files: curr_tmp_data_diff_U = tmp_data_diff_U
            if load_diff_files: curr_tmp_data_diff_V = tmp_data_diff_V
            
        #pdb.set_trace()
    
    #pdb.set_trace()
    '''



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
    

    '''
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

    if comp == 'hpc':
        pdb.set_trace()
        
    nc_time_origin = nctime[0].attrs['time_origin']
        
    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    if type(nctime.to_numpy()[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
        nctime_calendar_type = '360'
    else:
        nctime_calendar_type = 'greg'
    """
    nctime_calendar_type = 'greg'
    #pdb.set_trace()
    """

    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    #if type(nctime.to_numpy()[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
    if  nctime_calendar_type == '360':
        # if 360 days

        time_datetime_since_1970 = [ss.year + (ss.month-1)/12 + (ss.day-1)/360 for ss in nctime.to_numpy()]   
        time_datetime = time_datetime_since_1970
    else:
        # if gregorian

        
        sec_since_origin = [float(ii.data - np.datetime64(nc_time_origin))/1e9 for ii in nctime]
        time_datetime_cft = num2date(sec_since_origin,units = 'seconds since ' + nctime[0].attrs['time_origin'],calendar = 'gregorian') #nctime.calendar)

        time_datetime = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft])
        time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])

    #pdb.set_trace()
    # repeat if comparing two time series. 
    if subtracted_flist is not None:
        
        clim_sym = True
        tmp_data_diff = xarray.open_mfdataset(subtracted_flist ,combine='by_coords' )
        #pdb.set_trace()
        nav_lat_diff = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
        nav_lon_diff = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())

        nctime_diff = tmp_data_diff.variables['time_counter']

        nc_time_origin_diff = nctime_diff[0].attrs['time_origin']

        sec_since_origin_diff = [float(ii.data - np.datetime64(nc_time_origin_diff))/1e9 for ii in nctime_diff]
        time_datetime_cft_diff = num2date(sec_since_origin_diff,units = 'seconds since ' + nctime_diff[0].attrs['time_origin'],calendar = 'gregorian') #nctime.calendar)

        time_datetime_diff = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft_diff])
        time_datetime_since_1970_diff = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime_diff])
        
        # check both filessets have the same times
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
    
    #add "buttons"
    but_x0 = 0.01
    but_x1 = 0.06
    func_but_x1 = 0.99
    func_but_x0 = 0.94
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



    func_names_lst = ['Reset zoom', 'Zoom', 'Clim: Reset','Clim: Zoom','Clim: Expand','Clim: perc','Clim: normal', 'Clim: log','Surface', 'Near-Bed', 'Surface-Bed','Depth level','Save Figure', 'Quit']

    func_but_line_han,func_but_text_han = {},{}
    func_but_extent = {}
    #add button box
    for vi,funcname in enumerate(func_names_lst): 
        func_but_line_han[funcname] = clickax.plot([func_but_x0,func_but_x1,func_but_x1,func_but_x0,func_but_x0],0.9 - (np.array([0,0,but_dy,but_dy,0]) + vi*0.05),'k')
         #add button names
        func_but_text_han[funcname] = clickax.text((func_but_x0+func_but_x1)/2,0.9 - ((but_dy/2) + vi*0.05),funcname, ha = 'center', va = 'center')
    
        #note button extends (as in position.x0,x1, y0, y1)
        func_but_extent[funcname] = [func_but_x0,func_but_x1,0.9 - (but_dy + vi*0.05),0.9 - (0 + vi*0.05)]
    #pdb.set_trace()  


    func_but_text_han['Depth level'].set_color('r')
    func_but_text_han['Clim: normal'].set_color('b')
    but_text_han[var].set_color('r')

    def indices_from_ginput_ax(clii,cljj):
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


                # what do the local coordiantes of the click mean in terms of the data to plot.
                # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.

                # if in map, covert lon lat to ii,jj
                if ai == 0:
                    
                    loni,latj= xlocval,ylocval
                    if config == 'amm7':
                        sel_ii = (np.abs(lon - loni)).argmin()
                        sel_jj = (np.abs(lat - latj)).argmin()
                    elif config == 'amm15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                        sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)
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
                    if config == 'amm7':
                        sel_ii = (np.abs(lon - loni)).argmin()
                    elif config == 'amm15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                    else:
                        print('config not supported:', config)
                        pdb.set_trace()
                    
                    
                elif ai in [2]:
                    # if in ns slice, change ew slice, and hov/time series
                    latj= xlocval
                    if config == 'amm7':
                        sel_jj = (np.abs(lat - latj)).argmin()
                    elif config == 'amm15':
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)
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
        
        ew_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,jj,:].load())
        if load_diff_files: ew_slice_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,jj,:].load())
        ew_slice_x =  nav_lon[jj,:]
        ew_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,:]
        return ew_slice_dat,ew_slice_x, ew_slice_y

    def reload_ns_data():              
        ns_slice_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,:,ii].load())
        if load_diff_files: ns_slice_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,:,ii].load())
        ns_slice_x =  nav_lat[:,ii]
        ns_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,:,ii]
        return ns_slice_dat,ns_slice_x, ns_slice_y
                    
    def reload_hov_data():                
        hov_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,:,jj,ii].load()).T
        if load_diff_files: hov_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,:,jj,ii].load()).T
        hov_x = time_datetime
        hov_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,ii]
        return hov_dat,hov_x,hov_y

    def reload_ew_data_derived_var():
        
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        ew_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,jj,:].load())
        ew_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,jj,:].load())
        if load_diff_files: ew_slice_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,jj,:].load())
        if load_diff_files: ew_slice_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,jj,:].load())
        ew_slice_dat = np.sqrt(ew_slice_dat_U**2 + ew_slice_dat_V**2)
        ew_slice_x =  nav_lon[jj,:]
        ew_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,:]
        return ew_slice_dat,ew_slice_x, ew_slice_y

    def reload_ns_data_derived_var():              
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        ns_slice_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,:,ii].load())
        ns_slice_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,:,ii].load())
        if load_diff_files: ns_slice_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,:,ii].load())
        if load_diff_files: ns_slice_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,:,ii].load())
        ns_slice_dat = np.sqrt(ns_slice_dat_U**2 + ns_slice_dat_V**2)
        ns_slice_x =  nav_lat[:,ii]
        ns_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,:,ii]
        return ns_slice_dat,ns_slice_x, ns_slice_y
                    
    def reload_hov_data_derived_var():                
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        hov_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][:,:,jj,ii].load()).T
        hov_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][:,:,jj,ii].load()).T
        if load_diff_files: hov_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][:,:,jj,ii].load()).T
        if load_diff_files: hov_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][:,:,jj,ii].load()).T
        hov_dat = np.sqrt(hov_dat_U**2 + hov_dat_V**2)
        hov_x = time_datetime
        hov_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,ii]
        return hov_dat,hov_x,hov_y



    def reload_map_data_derived_var():
        tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
        if z_meth == 'z_slice':
            if var_dim[var] == 4:
                #pdb.set_trace()
                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
                if load_diff_files: map_dat_3d_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                if load_diff_files: map_dat_3d_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
                map_dat_3d = np.sqrt(map_dat_3d_U**2 + map_dat_3d_V**2)

                if zz not in interp1d_wgtT.keys(): interp1d_wgtT[zz] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,:,:],zz)
                map_dat =  interp1dmat_wgt(map_dat_3d,interp1d_wgtT[zz])
            
            elif var_dim[var] == 3:
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                if load_diff_files: map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                if load_diff_files: map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)

        elif z_meth in ['ss','nb','df']:

            if var_dim[var] == 4:

                map_dat_3d_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,:,::thin,::thin].load())
                map_dat_3d_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,:,::thin,::thin].load())
                map_dat_3d = np.sqrt(map_dat_3d_U**2 + map_dat_3d_V**2)
                map_dat_ss = map_dat_3d[0,:,:]
                map_dat_nb = np.ma.array(extract_nb(map_dat_3d[:,:,:],nbind),mask = tmask[0,:,:])
                if z_meth == 'ss': map_dat = map_dat_ss
                if z_meth == 'nb': map_dat = map_dat_nb
                if z_meth == 'df': map_dat = map_dat_ss - map_dat_nb
            elif var_dim[var] == 3:
                #map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                if load_diff_files: map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                if load_diff_files: map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        elif z_meth == 'z_index':
            if var_dim[var] == 4:
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,zz,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,zz,::thin,::thin].load())
                map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
            elif var_dim[var] == 3:
                #map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                map_dat_U = np.ma.masked_invalid(curr_tmp_data_U.variables[tmp_var_U][ti,::thin,::thin].load())
                map_dat_V = np.ma.masked_invalid(curr_tmp_data_V.variables[tmp_var_V][ti,::thin,::thin].load())
                if load_diff_files: map_dat_U -= np.ma.masked_invalid(curr_tmp_data_diff_U.variables[tmp_var_U][ti,::thin,::thin].load())
                if load_diff_files: map_dat_V -= np.ma.masked_invalid(curr_tmp_data_diff_V.variables[tmp_var_V][ti,::thin,::thin].load())
                map_dat = np.sqrt(map_dat_U**2 + map_dat_V**2)
        else:
            print('z_meth not supported:',z_meth)
            pdb.set_trace()

        map_x = nav_lon
        map_y = nav_lat
        
        return map_dat,map_x,map_y
                



    def reload_map_data():
        '''
        if var_grid[var] == 'T':
            curr_tmp_data = tmp_data
            if load_diff_files: curr_tmp_data_diff = tmp_data_diff
        elif var_grid[var] == 'U':
            curr_tmp_data = tmp_data_U
            if load_diff_files: curr_tmp_data_diff = tmp_data_diff_U
        elif var_grid[var] == 'V':
            curr_tmp_data = tmp_data_V
            if load_diff_files: curr_tmp_data_diff = tmp_data_diff_V
        else:
            print('grid dict error')
            pdb.set_trace()
        '''
        if z_meth == 'z_slice':
            if var_dim[var] == 4:
                map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
                if load_diff_files: map_dat_3d -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())


                if zz not in interp1d_wgtT.keys(): interp1d_wgtT[zz] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,:,:],zz)
                map_dat =  interp1dmat_wgt(map_dat_3d,interp1d_wgtT[zz])
            
            elif var_dim[var] == 3:
                map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                if load_diff_files: map_dat -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,::thin,::thin].load())

        elif z_meth in ['ss','nb','df']:

            if var_dim[var] == 4:

                map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,:,::thin,::thin].load())
                if load_diff_files: map_dat_3d -= np.ma.masked_invalid(curr_tmp_data_diff.variables[var][ti,:,::thin,::thin].load())
                map_dat_ss = map_dat_3d[0,:,:]
                map_dat_nb = np.ma.array(extract_nb(map_dat_3d[:,:,:],nbind),mask = tmask[0,:,:])
                if z_meth == 'ss': map_dat = map_dat_ss
                if z_meth == 'nb': map_dat = map_dat_nb
                if z_meth == 'df': map_dat = map_dat_ss - map_dat_nb
            elif var_dim[var] == 3:
                #map_dat_3d = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
                map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
        elif z_meth == 'z_index':
            if var_dim[var] == 4:
                map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,zz,::thin,::thin].load())
            elif var_dim[var] == 3:
                map_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][ti,::thin,::thin].load())
        else:
            print('z_meth not supported:',z_meth)
            pdb.set_trace()

        map_x = nav_lon
        map_y = nav_lat
        
        return map_dat,map_x,map_y
                


    '''

    def reload_map_data():
        

        if z_meth == 'z_slice':
            if var_dim[var] == 4:
                map_dat_3d = np.ma.masked_invalid(tmp_data.variables[var][ti,:,::thin,::thin].load())
                if load_diff_files: map_dat_3d -= np.ma.masked_invalid(tmp_data_diff.variables[var][ti,:,::thin,::thin].load())


                if zz not in interp1d_wgtT.keys(): interp1d_wgtT[zz] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,:,:],zz)
                map_dat =  interp1dmat_wgt(map_dat_3d,interp1d_wgtT[zz])
            
            elif var_dim[var] == 3:
                map_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,::thin,::thin].load())
                if load_diff_files: map_dat -= np.ma.masked_invalid(tmp_data_diff.variables[var][ti,::thin,::thin].load())

        elif z_meth in ['ss','nb','df']:

            if var_dim[var] == 4:

                map_dat_3d = np.ma.masked_invalid(tmp_data.variables[var][ti,:,::thin,::thin].load())
                map_dat_ss = map_dat_3d[0,:,:]
                map_dat_nb = np.ma.array(extract_nb(map_dat_3d[:,:,:],nbind),mask = tmask[0,:,:])
                if z_meth == 'ss': map_dat = map_dat_ss
                if z_meth == 'nb': map_dat = map_dat_nb
                if z_meth == 'df': map_dat = map_dat_ss - map_dat_nb
            elif var_dim[var] == 3:
                map_dat_3d = np.ma.masked_invalid(tmp_data.variables[var][ti,::thin,::thin].load())
        elif z_meth == 'z_index':
            if var_dim[var] == 4:
                map_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,zz,::thin,::thin].load())
            elif var_dim[var] == 3:
                map_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,::thin,::thin].load())
        else:
            print('z_meth not supported:',z_meth)
            pdb.set_trace()

        map_x = nav_lon
        map_y = nav_lat
        
        return map_dat,map_x,map_y
    '''


    '''
    def reload_ts_data():

        if z_meth == 'z_slice':
            if var_dim[var] == 4:            
                ts_dat = np.ma.masked_invalid(tmp_data.variables[var][:,zi,jj,ii].load())
                if load_diff_files:
                    ts_dat_1 = np.ma.masked_invalid(tmp_data.variables[var][:,zi,jj,ii].load())
                    ts_dat_2 = np.ma.masked_invalid(tmp_data_diff.variables[var][:,zi,jj,ii].load())
            elif var_dim[var] == 3:
                ts_dat = np.ma.masked_invalid(tmp_data.variables[var][:,jj,ii].load())
                if load_diff_files:
                    ts_dat_1 = np.ma.masked_invalid(tmp_data.variables[var][:,jj,ii].load())
                    ts_dat_2 = np.ma.masked_invalid(tmp_data_diff.variables[var][:,jj,ii].load())
        elif z_meth in ['ss','nb','df']:
            pdb.set_trace()
    
            if var_dim[var] == 4:            
                ss_ts_dat = np.ma.masked_invalid(tmp_data.variables[var][:,0,jj,ii].load())
                pdb.set_trace()
                nb_ts_dat = np.ma.array(extract_nb(tmp_data.variables[var][:,:,jj,ii].load(),nbind[:,jj,ii]))

                if load_diff_files:
                    ss_ts_dat_2 = np.ma.masked_invalid(tmp_data_diff.variables[var][:,0,jj,ii].load())
            elif var_dim[var] == 3:
                ts_dat = np.ma.masked_invalid(tmp_data.variables[var][:,jj,ii].load())
                if load_diff_files:
                    ts_dat_1 = np.ma.masked_invalid(tmp_data.variables[var][:,jj,ii].load())
                    ts_dat_2 = np.ma.masked_invalid(tmp_data_diff.variables[var][:,jj,ii].load())



        ts_x = time_datetime
        #return ts_x,ts_dat_1,ts_dat_2
    '''

    #get the current xlim (default to None??)
    cur_xlim = xlim
    cur_ylim = ylim
    # only load data when needed
    reload_map, reload_ew, reload_ns, reload_hov, reload_ts = True,True,True,True,True
    if z_meth_default == 'z_slice':
        interp1d_wgtT = {}
        interp1d_wgtT[0] = interp1dmat_create_weight(rootgrp_gdept.variables['gdept_0'][0,:,:,:],0)
    #pdb.set_trace()
    # loop
    while ii is not None:
        # try, exit on error
        #try:
        if True: 
            # extract plotting data (when needed), and subtract off difference files if necessary.



            if var_grid[var] == 'T':
                curr_tmp_data = tmp_data
                if load_diff_files: curr_tmp_data_diff = tmp_data_diff
            elif var_grid[var] == 'U':
                curr_tmp_data = tmp_data_U
                if load_diff_files: curr_tmp_data_diff = tmp_data_diff_U
            elif var_grid[var] == 'V':
                curr_tmp_data = tmp_data_V
                if load_diff_files: curr_tmp_data_diff = tmp_data_diff_V
            elif var_grid[var] == 'UV':
                curr_tmp_data_U = tmp_data_U
                curr_tmp_data_V = tmp_data_V
                if load_diff_files: curr_tmp_data_diff_U = tmp_data_diff_U
                if load_diff_files: curr_tmp_data_diff_V = tmp_data_diff_V
            else:
                print('grid dict error')
                pdb.set_trace()


            if reload_map:
                if var in deriv_var:
                    map_dat,map_x,map_y = reload_map_data_derived_var()
                else:
                    map_dat,map_x,map_y = reload_map_data()
                reload_map = False
            if reload_ew:
                if var_dim[var] == 4:
                    
                    if var in deriv_var:
                        ew_slice_dat,ew_slice_x, ew_slice_y = reload_ew_data_derived_var()
                    else:
                        ew_slice_dat,ew_slice_x, ew_slice_y = reload_ew_data()
                reload_ew = False
            if reload_ns:
                if var_dim[var] == 4:
                    if var in deriv_var:
                        ns_slice_dat,ns_slice_x, ns_slice_y = reload_ns_data_derived_var()    
                    else:
                        ns_slice_dat,ns_slice_x, ns_slice_y = reload_ns_data()                    
                reload_ns = False
            if reload_hov:
                if var_dim[var] == 4:
                    if var in deriv_var:
                        hov_dat,hov_x,hov_y = reload_hov_data_derived_var()
                    else:
                        hov_dat,hov_x,hov_y = reload_hov_data()


                reload_hov = False
            if reload_ts:
                if var_grid[var] != 'UV':
                    if var_dim[var] == 4:
                    
                        ts_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,zi,jj,ii].load())
                        if load_diff_files:
                            ts_dat_1 = np.ma.masked_invalid(curr_tmp_data.variables[var][:,zi,jj,ii].load())
                            ts_dat_2 = np.ma.masked_invalid(curr_tmp_data_diff.variables[var][:,zi,jj,ii].load())
                    elif var_dim[var] == 3:
                        ts_dat = np.ma.masked_invalid(curr_tmp_data.variables[var][:,jj,ii].load())
                        if load_diff_files:
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
                        if load_diff_files:
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
                
            
            #plot data
            pax = []
            #pdb.set_trace()
            pax.append(ax[0].pcolormesh(map_x,map_y,map_dat,cmap = curr_cmap,norm = climnorm))
            if var_dim[var] == 4:
                pax.append(ax[1].pcolormesh(ew_slice_x,ew_slice_y,ew_slice_dat,cmap = curr_cmap,norm = climnorm))
                pax.append(ax[2].pcolormesh(ns_slice_x,ns_slice_y,ns_slice_dat,cmap = curr_cmap,norm = climnorm))
                pax.append(ax[3].pcolormesh(hov_x,hov_y,hov_dat,cmap = curr_cmap,norm = climnorm))
            tsax = ax[4].plot(ts_x,ts_dat,'r')
            # add variable name as title - maybe better as a button color chnage?
            ax[0].set_title('%s (%i, %i, %i, %i) '%(var,ii,jj,zz,ti))
            
            # add colorbars
            #print('add colorbars')
            cax = []      
            if var_dim[var] == 4:  
                for ai in [0,1,2,3]: cax.append(plt.colorbar(pax[ai], ax = ax[ai]))
            elif var_dim[var] == 3:
                for ai in [0]: cax.append(plt.colorbar(pax[ai], ax = ax[ai]))
            #print('added colorbars')
            
            # apply xlim/ylim if keyword set
            #if xlim is not None:ax[0].set_xlim(cur_xlim)
            #if ylim is not None:ax[0].set_ylim(cur_ylim)
            #if ylim is not None:ax[1].set_xlim(cur_xlim)
            #if xlim is not None:ax[2].set_xlim(cur_ylim)
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
            
            if clim is None:
                for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)
                #When using the log scale, the colour set_clim seems linked, so all panels get set to the limits of the final set_perc_clim_pcolor call..
                #   therefore repeat set_perc_clim_pcolor of the map, so the hovmuller colour limit is not the final one. 
                set_perc_clim_pcolor_in_region(5,95, ax = ax[0],sym = clim_sym)
                '''
                if climnorm is None:
                    for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)
                else:
                    #pdb.set_trace()
                    tmp_map_clim = np.log10(np.percentile(10**map_dat[map_dat.mask == False],(5,95)))
                    tmp_ew_slice_clim = np.log10(np.percentile(10**ew_slice_dat[ew_slice_dat.mask == False],(5,95)))
                    tmp_ns_slice_clim = np.log10(np.percentile(10**ns_slice_dat[ns_slice_dat.mask == False],(5,95)))
                    tmp_hov_clim = np.log10(np.percentile(10**hov_dat[hov_dat.mask == False],(5,95)))
                    tmp_map_clim,tmp_ew_slice_clim,tmp_ns_slice_clim,tmp_hov_clim
                    print(ax)
                    for tmpax in ax:print(id(tmpax))
                    set_clim_pcolor(tmp_map_clim, ax = ax[0])
                    set_clim_pcolor(tmp_ew_slice_clim, ax = ax[1])
                    set_clim_pcolor(tmp_ns_slice_clim, ax = ax[2])
                    set_clim_pcolor(tmp_hov_clim, ax = ax[3])
                    set_clim_pcolor(tmp_map_clim, ax = ax[0])
                    for tmpax in ax[:-1]:print(id(tmpax),get_clim_pcolor(ax = tmpax)) 
                    set_clim_pcolor(tmp_map_clim, ax = ax[0])
                    set_clim_pcolor(tmp_ew_slice_clim, ax = ax[1])
                    set_clim_pcolor(tmp_ns_slice_clim, ax = ax[2])
                    set_clim_pcolor(tmp_hov_clim, ax = ax[3])
                    set_clim_pcolor(tmp_map_clim, ax = ax[0])
                '''
            else:
                for tmpax in ax[:-1]:set_clim_pcolor((clim), ax = tmpax)
        
            #for tmpax in ax[:-1]:print('updated clim',get_clim_pcolor(ax = tmpax))    
        
            #print('Have reset colour limits')
            
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

            
            # set current axes to hidden full screen axes for click interpretation
            plt.sca(clickax)
            
            #await click with ginput
            tmp = plt.ginput(1)
            if len(tmp) == 0: continue
            clii,cljj = tmp[0][0],tmp[0][1]

            #get click location, and current axis limits for ax[0], and set them
            # defunct? was trying to allow zooming
            cur_xlim = np.array(ax[0].get_xlim())
            cur_ylim = np.array(ax[0].get_ylim())

            ax[0].set_xlim(cur_xlim)
            ax[0].set_ylim(cur_ylim)

            #find clicked axes:
            is_in_axes = False
            
            '''
            # moved into an inner function

            #pdb.set_trace()
            #cycle through plotting axes, and see if clicked.
            for ai,tmpax in enumerate(ax): 
                #axes position (extent)
                tmppos =  tmpax.get_position()
                # was click within extent
                if (clii >= tmppos.x0) & (clii <= tmppos.x1) & (cljj >= tmppos.y0) & (cljj <= tmppos.y1):
                    #if so:
                    is_in_axes = True

                    #convert figure coordinate of click, into location with the axes, using data coordinates
                    clxlim = np.array(tmpax.get_xlim())
                    clylim = np.array(tmpax.get_ylim())
                    normxloc = (clii - tmppos.x0 ) / (tmppos.x1 - tmppos.x0)
                    normyloc = (cljj - tmppos.y0 ) / (tmppos.y1 - tmppos.y0)
                    xlocval = normxloc*clxlim.ptp() + clxlim.min()
                    ylocval = normyloc*clylim.ptp() + clylim.min()
                    #print(clii,clxlim,normxloc,xlocval)
                    #print(cljj,clylim,normyloc,ylocval)


                    # what do the local coordiantes of the click mean in terms of the data to plot.
                    # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.

                    # if in map, covert lon lat to ii,jj
                    if ai == 0:
                        loni,latj= xlocval,ylocval
                        if config == 'amm7':
                            ii = (np.abs(lon - loni)).argmin()
                            jj = (np.abs(lat - latj)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                            jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)
                        # and reload slices, and hovmuller/time series
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True

                    elif ai in [1]: 
                        # if in ew slice, change ns slice, and hov/time series
                        loni= xlocval
                        if config == 'amm7':
                            ii = (np.abs(lon - loni)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                        
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True
                        
                    elif ai in [2]:
                        # if in ns slice, change ew slice, and hov/time series
                        latj= xlocval
                        if config == 'amm7':
                            jj = (np.abs(lat - latj)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)

                        reload_ew = True
                        reload_hov = True
                        reload_ts = True

                    elif ai in [3]:
                        # if in hov/time series, change map, and slices

                        # re calculate depth values, as y scale reversed, 
                        zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )

                        
                        reload_map = True
                        reload_ts = True

                    elif ai in [4]:
                        # if in hov/time series, change map, and slices
                        ti = np.abs(xlocval - time_datetime_since_1970).argmin()
                        
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                    else:
                        return
                        pdb.set_trace()
            '''
            # convert the mouse click into data indices, and report which axes was clicked
            sel_ax,sel_ii,sel_jj,sel_ti,sel_zz = indices_from_ginput_ax(clii,cljj)

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
                        
                        #convert clicks to data indices
                        zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz = indices_from_ginput_ax(tmpzoom[0][0],tmpzoom[0][1])
                        zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz = indices_from_ginput_ax(tmpzoom[1][0],tmpzoom[1][1])
                        #print(zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz)
                        #print(zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz)
                        #print(cur_xlim)
                        #print(cur_ylim)
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
                        '''
                        pdb.set_trace()
                        #convert clicks to data indices
                        czoom0_ax,czoom0_ii,czoom0_jj = indices_from_ginput_cax(tmpczoom[0][0],tmpczoom[0][1])
                        czoom1_ax,czoom1_ii,czoom1_jj = indices_from_ginput_cax(tmpczoom[1][0],tmpczoom[1][1])

                        # if both clicks in main axes, use clicks for the new x and ylims
                        if (czoom0_ax is not None) & (czoom1_ax is not None):
                            if czoom0_ax == czoom1_ax:
                                clim = [czoom0_jj,czoom1_jj]
                        '''

                    elif but_name == 'Clim: Expand': 
                        clim = np.array(get_clim_pcolor(ax = ax[0]))
                        if climnorm is None:
                            clim = np.array([clim.mean() - clim.ptp(),clim.mean() + clim.ptp()])
                        else:
                            clim = np.log10(np.array([(10**clim).mean() - (10**clim).ptp(),(10**clim).mean() + (10**clim).ptp()]))
                        
                    
                    elif but_name == 'Clim: perc': 
                        clim = None

                    
                    elif but_name == 'Clim: log': 
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
                    elif but_name in [' ']:
                        func_but_text_han['Depth level'].set_color('k')
                        func_but_text_han['Surface'].set_color('k')
                        func_but_text_han['Near-Bed'].set_color('k')
                        func_but_text_han['Surface-Bed'].set_color('k')
                        func_but_text_han[but_name].set_color('r')
                        z_meth = z_meth_default    
                        reload_map = True
                        reload_ts = True
                    elif but_name in 'Save Figure':                        
                        fig.savefig('/home/h01/hadjt/workspace/python3/tmpfig_%i%i_%i_%i.png'%(ii,jj,ti,zz))
                    elif but_name in 'Quit':
                        return
                    else:
                        print(but_name)
                        pdb.set_trace()
                    print(clim)
                        
                        
            '''
            # if no axes, or buttons pressed, quite. 
            if is_in_axes == False:
                print('Clicked outside axes')
                return
                pdb.set_trace()
            '''        
            
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



            #fig.canvas.draw()
#            pdb.set_trace() # cs_line
        '''
        except:
            print('excepted',ii,jj,ti,zi)
            tmp_data.close()
            rootgrp_gdept.close()
            return
        '''


"""


def nemo_slice_slev(fname_lst, subtracted_flist = None,var = 'votemper',config = 'amm7', thin = 1,
    zlim_max = None,xlim = None, ylim = None, tlim = None, clim = None,
    ii = None, jj = None, ti = None, zi = None ):

    # default color map to use
    curr_cmap = None
    diff_cmap = cmocean.cm.balance

    # default initial indices
    if ii is None: ii = 120
    if jj is None: jj = 120
    if ti is None: ti = 0
    if zi is None: zi = 0


    #config version specific info - mainly grid, and lat/lon info
    if config == 'amm7':
        # depth grid file
        rootgrp_gdept = Dataset('/data/cr1/hadjt/data/reffiles/SSF/amm7.mesh_mask.nc', 'r', format='NETCDF4')
        # depth grid variable name
        zss = 'gdept'

        #grid lat lon
        lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
        lat = np.arange(40.066669,65+1/15.,1/15.)
        
    elif config == 'amm15':

        # depth grid file
        rootgrp_gdept = Dataset('/data/cr1/hadjt/data/reffiles/SSF/amm15.mesh_mask.nc', 'r', format='NETCDF4')
        # depth grid variable name
        zss = 'gdept_0'
        
        # grid lat lon rotation information
        lon_rotamm15,lat_rotamm15 = reduce_rotamm15_grid()

        dlon_rotamm15 = (np.diff(lon_rotamm15)).mean()
        dlat_rotamm15 = (np.diff(lat_rotamm15)).mean()
        nlon_rotamm15 = lon_rotamm15.size
        nlat_rotamm15 = lat_rotamm15.size

    # open file list with xarray
    tmp_data = xarray.open_mfdataset(fname_lst ,combine='by_coords') # , decode_cf=False)
    # load nav_lat and nav_lon
    nav_lat = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
    nav_lon = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())
    
    # what are the4d variable names, and how many are there?
    var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 4])
    nvar4d = var_4d_mat.size


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


    nc_time_origin = nctime[0].attrs['time_origin']


    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    if type(nctime.to_numpy()[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
        # if 360 days

        time_datetime_since_1970 = [ss.year + (ss.month-1)/12 + (ss.day-1)/360 for ss in nctime.to_numpy()]   
        time_datetime = time_datetime_since_1970
    else:
        # if gregorian


        sec_since_origin = [float(ii.data - np.datetime64(nc_time_origin))/1e9 for ii in nctime]
        time_datetime_cft = num2date(sec_since_origin,units = 'seconds since ' + nctime[0].attrs['time_origin'],calendar = 'gregorian') #nctime.calendar)

        time_datetime = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft])
        time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])

    load_diff_files = False
    # repeat if comparing two time series. 
    if subtracted_flist is not None:
        
        load_diff_files = True
        tmp_data_diff = xarray.open_mfdataset(subtracted_flist ,combine='by_coords' )
        #pdb.set_trace()
        nav_lat_diff = np.ma.masked_invalid(tmp_data.variables['nav_lat'][::thin,::thin].load())
        nav_lon_diff = np.ma.masked_invalid(tmp_data.variables['nav_lon'][::thin,::thin].load())

        nctime_diff = tmp_data_diff.variables['time_counter']

        nc_time_origin_diff = nctime_diff[0].attrs['time_origin']

        sec_since_origin_diff = [float(ii.data - np.datetime64(nc_time_origin_diff))/1e9 for ii in nctime_diff]
        time_datetime_cft_diff = num2date(sec_since_origin_diff,units = 'seconds since ' + nctime_diff[0].attrs['time_origin'],calendar = 'gregorian') #nctime.calendar)

        time_datetime_diff = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft_diff])
        time_datetime_since_1970_diff = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime_diff])
        
        # check both filessets have the same times
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
   

    ax = []
    pax = []

    fig = plt.figure()
    fig.suptitle('Interactive figure, click outside axes to quit', fontsize=20)
    fig.set_figheight(12)
    fig.set_figwidth(18) 
    # add axes
    ax.append(plt.subplot(1,2,1))
    ax.append(plt.subplot(4,2,2))
    ax.append(plt.subplot(4,2,4))
    ax.append(plt.subplot(4,2,6))
    ax.append(plt.subplot(4,2,8))

    #flip depth axes
    for tmpax in ax[1:]: tmpax.invert_yaxis()
    #use log depth scale, setiched off as often causes problems (clashes with hidden axes etc).
    #for tmpax in ax[1:]: tmpax.set_yscale('log')

    # add hidden fill screen axes 
    clickax = fig.add_axes([0,0,1,1], frameon=False)
    clickax.axis('off')
    
    #add "buttons"
    but_x0 = 0.01
    but_x1 = 0.05
    but_dy = 0.04
    but_ysp = 0.01 
    but_line_han,but_text_han = {},{}
    #add button box
    for vi,var_4d in enumerate(var_4d_mat): but_line_han[var_4d] = clickax.plot([but_x0,but_x1,but_x1,but_x0,but_x0],0.9 - (np.array([0,0,but_dy,but_dy,0]) + vi*0.05),'k')
    #add button names
    for vi,var_4d in enumerate(var_4d_mat): but_text_han[var_4d] = clickax.text((but_x0+but_x1)/2,0.9 - ((but_dy/2) + vi*0.05),var_4d, ha = 'center', va = 'center')
    clickax.axis([0,1,0,1])
    
    #note button extends (as in position.x0,x1, y0, y1)
    but_extent = {}
    for vi,var_4d in enumerate(var_4d_mat): but_extent[var_4d] = [but_x0,but_x1,0.9 - (but_dy + vi*0.05),0.9 - (0 + vi*0.05)]
    #pdb.set_trace()  

    #get the current xlim (default to None??)
    cur_xlim = xlim
    cur_ylim = ylim

    # only load data when needed
    reload_map, reload_ew, reload_ns, reload_hov, reload_ts = True,True,True,True,True

    # loop
    while ii is not None:
        # try, exit on error
        try:
        #if True: 
            # extract plotting data (when needed), and subtract off difference files if necessary.
            if reload_map:
                map_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,zi,::thin,::thin].load())
                if load_diff_files: map_dat -= np.ma.masked_invalid(tmp_data_diff.variables[var][ti,zi,::thin,::thin].load())
                map_x = nav_lon
                map_y = nav_lat
                reload_map = False
            if reload_ew:
                ew_slice_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,:,jj,:].load())
                if load_diff_files: ew_slice_dat -= np.ma.masked_invalid(tmp_data_diff.variables[var][ti,:,jj,:].load())
                ew_slice_x =  nav_lon[jj,:]
                ew_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,:]
                reload_ew = False
            if reload_ns:
                ns_slice_dat = np.ma.masked_invalid(tmp_data.variables[var][ti,:,:,ii].load())
                if load_diff_files: ns_slice_dat -= np.ma.masked_invalid(tmp_data_diff.variables[var][ti,:,:,ii].load())
                ns_slice_x =  nav_lat[:,ii]
                ns_slice_y =  rootgrp_gdept.variables['gdept_0'][0,:,:,ii]
                reload_ns = False
            if reload_hov:
                hov_dat = np.ma.masked_invalid(tmp_data.variables[var][:,:,jj,ii].load()).T
                if load_diff_files: hov_dat -= np.ma.masked_invalid(tmp_data_diff.variables[var][:,:,jj,ii].load()).T
                hov_x = time_datetime
                hov_y =  rootgrp_gdept.variables['gdept_0'][0,:,jj,ii]
                reload_hov = False
            if reload_ts:
                ts_dat = hov_dat[zi,:]
                if load_diff_files:
                    ts_dat_1 = np.ma.masked_invalid(tmp_data.variables[var][:,zi,jj,ii].load())
                    ts_dat_2 = np.ma.masked_invalid(tmp_data_diff.variables[var][:,zi,jj,ii].load())
                ts_x = time_datetime
                reload_ts = False
            
            
            #plot data
            pax = []
            pax.append(ax[0].pcolormesh(map_x,map_y,map_dat,cmap = curr_cmap))
            pax.append(ax[1].pcolormesh(ew_slice_x,ew_slice_y,ew_slice_dat,cmap = curr_cmap))
            pax.append(ax[2].pcolormesh(ns_slice_x,ns_slice_y,ns_slice_dat,cmap = curr_cmap))
            pax.append(ax[3].pcolormesh(hov_x,hov_y,hov_dat,cmap = curr_cmap))
            tsax = ax[4].plot(ts_x,ts_dat,'r')
            # add variable name as title - maybe better as a button color chnage?
            ax[0].set_title(var)
            
            # add colorbars
            cax = []        
            for ai in [0,1,2,3]: cax.append(plt.colorbar(pax[ai], ax = ax[ai]))
            
            # set minimum depth if keyword set
            xlim_min = 1
            if zlim_max == None:
                ax[1].set_ylim([ew_slice_y.max(),xlim_min])
                ax[2].set_ylim([ns_slice_y.max(),xlim_min])
                ax[3].set_ylim([hov_y.max(),xlim_min])
            else:
                ax[1].set_ylim([zlim_max,xlim_min])
                ax[2].set_ylim([zlim_max,xlim_min])
                ax[3].set_ylim([np.minimum(zlim_max,hov_y.max()),xlim_min])

            # apply xlim/ylim if keyword set
            if xlim is not None:ax[0].set_xlim(cur_xlim)
            if ylim is not None:ax[0].set_ylim(cur_ylim)
            if ylim is not None:ax[1].set_xlim(cur_xlim)
            if xlim is not None:ax[2].set_xlim(cur_ylim)
            if tlim is not None:ax[3].set_xlim(tlim)
            if tlim is not None:ax[4].set_xlim(tlim)
            
            #reset ylim to time series to data min max
            ax[4].set_ylim(ts_dat.min(),ts_dat.max())

            # if no keyword clim, use 5th and 95th percentile of data
            if clim is None:
                for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax)
            else:
                for tmpax in ax[:-1]:set_clim_pcolor((clim), ax = tmpax)
        
            
            ## add lines to show current point. 
            # currently using axhline, axvline, not idea of rotated grid (amm15)
            cs_line = []
            cs_line.append(ax[0].axhline(nav_lat[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[0].axvline(nav_lon[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[1].axvline(nav_lon[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[2].axvline(nav_lat[jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[3].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))
            cs_line.append(ax[4].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))

            # set current axes to hidden full screen axes for click interpretation
            plt.sca(clickax)
            
            #await click with ginput
            tmp = plt.ginput(1)
            if len(tmp) == 0: continue
            clii,cljj = tmp[0][0],tmp[0][1]

            #get click location, and current axis limits for ax[0], and set them
            # defunct? was trying to allow zooming
            cur_xlim = np.array(ax[0].get_xlim())
            cur_ylim = np.array(ax[0].get_ylim())

            ax[0].set_xlim(cur_xlim)
            ax[0].set_ylim(cur_ylim)

            #find clicked axes:
            is_in_axes = False
            
            #cycle through plotting axes, and see if clicked.
            for ai,tmpax in enumerate(ax): 
                #axes position (extent)
                tmppos =  tmpax.get_position()
                # was click within extent
                if (clii >= tmppos.x0) & (clii <= tmppos.x1) & (cljj >= tmppos.y0) & (cljj <= tmppos.y1):
                    #if so:
                    is_in_axes = True

                    #convert figure coordinate of click, into location with the axes, using data coordinates
                    clxlim = np.array(tmpax.get_xlim())
                    clylim = np.array(tmpax.get_ylim())
                    normxloc = (clii - tmppos.x0 ) / (tmppos.x1 - tmppos.x0)
                    normyloc = (cljj - tmppos.y0 ) / (tmppos.y1 - tmppos.y0)
                    xlocval = normxloc*clxlim.ptp() + clxlim.min()
                    ylocval = normyloc*clylim.ptp() + clylim.min()
                    #print(clii,clxlim,normxloc,xlocval)
                    #print(cljj,clylim,normyloc,ylocval)


                    # what do the local coordiantes of the click mean in terms of the data to plot.
                    # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.

                    # if in map, covert lon lat to ii,jj
                    if ai == 0:
                        loni,latj= xlocval,ylocval
                        if config == 'amm7':
                            ii = (np.abs(lon - loni)).argmin()
                            jj = (np.abs(lat - latj)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                            jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)
                        # and reload slices, and hovmuller/time series
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True

                    elif ai in [1]: 
                        # if in ew slice, change ns slice, and hov/time series
                        loni= xlocval
                        if config == 'amm7':
                            ii = (np.abs(lon - loni)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
                        
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True
                        
                    elif ai in [2]:
                        # if in ns slice, change ew slice, and hov/time series
                        latj= xlocval
                        if config == 'amm7':
                            jj = (np.abs(lat - latj)).argmin()
                        elif config == 'amm15':
                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                            jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)

                        reload_ew = True
                        reload_hov = True
                        reload_ts = True

                    elif ai in [3,4]:
                        # if in hov/time series, change map, and slices
                        ti = np.abs(xlocval - time_datetime_since_1970).argmin()
                        
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                    else:
                        return
                        pdb.set_trace()

            # if in button, change variables. 

            for but_name in but_extent.keys():
                
                but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = but_extent[but_name]
                if (clii >= but_pos_x0) & (clii <= but_pos_x1) & (cljj >= but_pos_y0) & (cljj <= but_pos_y1):
                    is_in_axes = True
                    if but_name in var_4d_mat:
                        var = but_name
                        reload_map = True
                        reload_ew = True
                        reload_ns = True
                        reload_hov = True
                        reload_ts = True
                        
                        
                    
            # if no axes, or buttons pressed, quite. 
            if is_in_axes == False:
                print('Clicked outside axes')
                return
                pdb.set_trace()
                    
            
            plt.sca(ax[0])
                    


            print("selected ii = %i,jj = %i,ti = %i,zi = %i, var = '%s'"%(ii,jj, ti, zi,var))
            # after selected indices and vareiabels, delete plots, ready for next cycle
        
            for tmp_cax in cax:tmp_cax.remove()
            for tmp_pax in pax:tmp_pax.remove()
            for tmp_cs_line in cs_line:tmp_cs_line.remove()
            rem_loc = tsax.pop(0)
            rem_loc.remove()
        except:
            print('excepted',ii,jj,ti,zi)
            tmp_data.close()
            rootgrp_gdept.close()
            return
"""
def main():




    if sys.argv.__len__() > 1:
        nvargin = sys.argv.__len__()
        python_code = sys.argv[0]
        print('nvargin',nvargin)

        config = sys.argv[1]
        zlim_max = sys.argv[2]
        if zlim_max.upper() == 'NONE':
            zlim_max = None
        else:
            zlim_max = int(zlim_max)
        fname_lst=sys.argv[3]
        #pdb.set_trace()
        nemo_slice_zlev(fname_lst,zlim_max = zlim_max, config = config) 
    
        exit()




    '''
    module load scitools/default-current
    python NEMO_nc_slevel_viewer_dev.py amm7 200 '/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_20231*_00.-36.nc'

    python /home/d05/hadjt/scripts/python/NEMO_nc_slevel_viewer.py amm7 200 /critical/opfc/suites-oper/foam_amm7/share/cycle/20231*T0000Z/level0/prodm_op_am-dm.gridT_*_00.-36.nc
    python /home/d05/hadjt/scripts/python/NEMO_nc_slevel_viewer.py amm7 200 '/critical/opfc/suites-oper/foam_amm7/share/cycle/$(date +%Y%m%d)T0000Z/level0/prodm_op_am-dm.gridT_*_00.???.nc'


    
    module load scitools/default-current
    python /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer.py amm7 200 '/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_202311??_00.-36.nc'


    module load scitools/default-current
    python /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer.py amm7 200 '/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_202311??_00.-36.nc'

    module load scitools/default-current
    python /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer.py ORCA025ext None '/project/oceanver/monitoring/input/cplnwp_opfc/diagnostics/run_20231204/cplnwp.mersea.grid_T.nc'



    fname_lst = glob.glob('/project/oceanver/monitoring/input/cplnwp_opfc/diagnostics/run_20231204/cplnwp.mersea.grid_T.nc')
    var = 'votemper'
    nemo_slice_zlev(fname_lst,config = 'orca025ext', xlim = [-15,20], ylim = [40,70])    
    


    '''

    if comp == 'hpc':
         fname_lst = glob.glob('/critical/opfc/suites-oper/foam_amm7/share/cycle/20231*T0000Z/level0/prodm_op_am-dm.gridT_*_00.-36.nc')
         nemo_slice_zlev(fname_lst, config = 'amm7',zlim_max = 200) # , xlim = [-5,10], ylim = [50,60]) 
   
    

    '''


/scratch/hadjt/SSF/tmp/LBC_implementation_shock
[hadjt@vld054 LBC_implementation_shock ]$ l
total 28
drwxr-xr-x.  2 hadjt users 4096 Nov 24 12:34 mi-bc709_orca12
drwxr-xr-x.  2 hadjt users 4096 Nov 24 12:44 mi-bc710_orca12
drwxr-xr-x.  2 hadjt users 4096 Nov 24 12:59 amm15_diff
drwxr-xr-x.  2 hadjt users 4096 Nov 29 10:09 amm7_diff
drwxr-xr-x.  7 hadjt users 2048 Nov 30 10:07 ..
drwxr-xr-x. 10 hadjt users 2048 Dec 13 09:21 .
drwxr-xr-x.  2 hadjt users 8192 Dec 13 09:46 OpSys_amm15
drwxr-xr-x.  2 hadjt users 4096 Dec 13 09:50 OpSys_amm7
drwxr-xr-x.  2 hadjt users 2048 Dec 13 09:50 mi-bc709_preorca
drwxr-xr-x.  2 hadjt users 2048 Dec 13 09:51 mi-bc710_preorca


    '''

    fname_type = 'am-dm'
    fname_lst_amm7_opsys = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm7/prodm_op_%s.gridT_20231130_00.-36.nc'%(fname_type)) + glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm7/prodm_op_%s.gridT_2023120[12]_00.-36.nc'%(fname_type)) 
    fname_lst_amm7_trial = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/mi-bc709_preorca/prodm_op_%s.gridT_*_00.-36.nc'%fname_type)

    fname_lst_amm15_opsys = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm15/prodm_op_%s.gridT_20231130_00.-36.nc'%(fname_type)) + glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm15/prodm_op_%s.gridT_2023120[12]_00.-36.nc'%(fname_type)) 
    fname_lst_amm15_trial = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/mi-bc710_preorca/prodm_op_%s.gridT_*_00.-36.nc'%fname_type)


    pdb.set_trace()


    #nemo_slice_zlev(fname_lst_amm7_opsys, config = 'amm7',zlim_max = 200) # , xlim = [-5,10], ylim = [50,60]) 
    #nemo_slice_zlev(fname_lst_amm7_trial, config = 'amm7',zlim_max = 200) # , xlim = [-5,10], ylim = [50,60]) 
    nemo_slice_zlev(fname_lst_amm7_opsys,subtracted_flist = fname_lst_amm7_trial, config = 'amm7',zlim_max = 200) # , xlim = [-5,10], ylim = [50,60]) 
    nemo_slice_zlev(fname_lst_amm15_opsys,subtracted_flist = fname_lst_amm15_trial, config = 'amm15',zlim_max = 200) # , xlim = [-5,10], ylim = [50,60]) 

    #i-bc709_preorca

    pdb.set_trace()

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_2023051?_00.-36.nc')
    fname_lst_U = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridU_2023051?_00.-36.nc')
    fname_lst_V = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridV_2023051?_00.-36.nc')
    nemo_slice_zlev(fname_lst, config = 'amm7',U_flist = fname_lst_U,V_flist = fname_lst_V ) # , xlim = [-5,10], ylim = [50,60]) 
    nemo_slice_zlev(fname_lst, config = 'amm7') # , xlim = [-5,10], ylim = [50,60]) 

    exit()
############################################################################

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridU_2023051?_00.-36.nc')
    nemo_slice_zlev(fname_lst, config = 'amm7',zlim_max = 75) # , xlim = [-5,10], ylim = [50,60]) 


    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_2023051?_00.-36.nc')
    fname_lst_diff = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21/prodm_op_am-dm.gridT_2023051?_00.-36.nc')
    var = 'votemper'  
    nemo_slice_zlev(fname_lst, var = var,config = 'amm7',zlim_max = 75) # , xlim = [-5,10], ylim = [50,60]) 
    nemo_slice_zlev(fname_lst,subtracted_flist = fname_lst_diff, var = var,config = 'amm7',zlim_max = 75, xlim = [-5,10], ylim = [50,60])  
    pdb.set_trace()   
    
    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm15/prodm_op_am-dm.gridT_20231121_00.-36.nc')

    var = 'votemper'  
    
    nemo_slice_zlev(fname_lst, var = var,config = 'amm15',zlim_max = 75, xlim = [-5,10], ylim = [50,60])   

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_202307??_00.-36.nc')
    #nemo_slice_zlev(fname_lst,config = 'amm7',zlim_max = 75, xlim = [-5,10], ylim = [50,60])    
    #pdb.set_trace()
    

    '''
    fname_lst = glob.glob('/scratch/hadjt/HCCP_UKCP_PPE/Results/tmp/HCCP_CO9_ap977_ar095_au084_r001i1p00000_02/2000*_Monthly3D_grid_T.nc')
    var = 'vosaline'
    nemo_slice_zlev(fname_lst, var = var,config = 'amm7',zlim_max = 200, xlim = [-5,15], ylim = [50,63])        
    pdb.set_trace()
    '''

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_2023*_00.-36.nc')

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_20231*_00.-36.nc')

    var = 'votemper'
    nemo_slice_zlev(fname_lst, var = var,config = 'amm7',zlim_max = 75) # , xlim = [-5,10], ylim = [50,60]) 
    pdb.set_trace()   
   

    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21_may23/prodm_op_am-dm.gridT_2023051?_00.-36.nc')
    fname_lst_diff = glob.glob('/scratch/hadjt/SSF/tmp/BGC_O2_blowup/OS45_bgc_fix_rerun21/prodm_op_am-dm.gridT_2023051?_00.-36.nc')
    var = 'votemper'
    nemo_slice_zlev(fname_lst, var = var,config = 'amm7',zlim_max = 75) # , xlim = [-5,10], ylim = [50,60]) 
    pdb.set_trace()   
    
    nemo_slice_zlev(fname_lst,subtracted_flist = fname_lst_diff, var = var,config = 'amm7',zlim_max = 75, xlim = [-5,10], ylim = [50,60])    
    nemo_slice_zlev(fname_lst,var = var,config = 'amm7',zlim_max = 75, xlim = [-5,10], ylim = [50,60])    
      
    pdb.set_trace()
   
    fname_lst = glob.glob('/scratch/hadjt/SSF/tmp/LBC_implementation_shock/OpSys_amm15/prodm_op_am-dm.gridT_20231121_00.-36.nc')

    var = 'votemper'  
    
    nemo_slice_zlev(fname_lst, var = var,config = 'amm15',zlim_max = 75, xlim = [-5,10], ylim = [50,60])   




if __name__ == "__main__":
    main()


