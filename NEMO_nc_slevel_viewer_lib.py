
import pdb,sys,os,cftime,socket

from datetime import datetime, timedelta

from netCDF4 import Dataset,num2date,stringtochar, chartostring

import numpy as np

import xarray
import csv

import matplotlib.pyplot as plt


from matplotlib.colors import LinearSegmentedColormap, ListedColormap



computername = socket.gethostname()
comp = 'linux'
if computername in ['xcel00','xcfl00']: comp = 'hpc'



RotNPole_lon = 177.5
RotNPole_lat = 37.5
RotSPole_lon = RotNPole_lon-180
RotSPole_lat = RotNPole_lat*-1
NP_coor = np.array([RotNPole_lon,RotNPole_lat])
SP_coor = np.array([RotSPole_lon,RotSPole_lat])



def load_nc_dims(tmp_data):
    # Find out the name of the lat, lon, depth and time dimension.
    # If a new model is used with a different dimension, you may need to add it here



    x_dim = 'x'
    y_dim = 'y'
    z_dim = 'deptht'
    t_dim = 'time_counter'
    #pdb.set_trace()

    '''	
    x_grid_T = 1238 ;
	y_grid_T = 1046 ;
	x_grid_U = 1238 ;
	y_grid_U = 1046 ;
	x_grid_V = 1238 ;
	y_grid_V = 1046 ;
	x_grid_T_inner = 1238 ;
	y_grid_T_inner = 1046 ;


    '''
    
    nc_dims = [ss for ss in tmp_data._dims.keys()]

    poss_zdims = ['depth','deptht','depthu','depthv','depthw','z', 'nc']
    poss_tdims = ['time_counter','time','t']
    poss_xdims = ['x','X','lon','ni','x_grid_T','x_grid_U','x_grid_V', 'lon','longitude','xbt','xbT','xbU','xbV']
    poss_ydims = ['y','Y','lat','nj','y_grid_T','y_grid_U','y_grid_V', 'lat','latitude','yb']


    '''
    WW3 doesn't have the right dimension, so needs a nemo grid first

    ncdump -h /data/scratch/frwave/wave_rolling_archive/amm15/amm15_2025062000.nc |head
    netcdf amm15_2025062000 {
    dimensions:
        time = UNLIMITED ; // (169 currently)
        seapoint = 394316 ;
    variables:

    
    '''
    
    if x_dim not in nc_dims: 
        x_dim_lst = [i for i in nc_dims if i in poss_xdims]
        if len(x_dim_lst)>0: 
            x_dim = x_dim_lst[0]
        else:
            x_dim = ''
    if y_dim not in nc_dims: 
        y_dim_lst = [i for i in nc_dims if i in poss_ydims]
        if len(y_dim_lst)>0: 
            y_dim = y_dim_lst[0]
        else:
            y_dim = ''
    if z_dim not in nc_dims: 
        z_dim_lst = [i for i in nc_dims if i in poss_zdims]
        if len(z_dim_lst)>0: 
            z_dim = z_dim_lst[0]
        else:
            z_dim = ''
    if t_dim not in nc_dims: 
        t_dim_lst = [i for i in nc_dims if i in poss_tdims]
        if len(t_dim_lst)>0: 
            t_dim = t_dim_lst[0]
        else:
            t_dim = ''
    return x_dim, y_dim, z_dim,t_dim


def load_nc_var_name_list(tmp_data,x_dim, y_dim, z_dim,t_dim):


    do_addtimedim = False
    # what are the4d variable names, and how many are there?
    #var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 4])

    var_4d_lst = [ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim, z_dim,y_dim, x_dim)]
    if do_addtimedim:
        var_4d_lst_notime = [ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (z_dim,y_dim, x_dim)]
        var_4d_mat = np.array(var_4d_lst + var_4d_lst_notime)
    else:
        var_4d_mat = np.array(var_4d_lst)
    nvar4d = var_4d_mat.size
    #pdb.set_trace()
    #var_3d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 3])
    var_3d_lst = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim, y_dim, x_dim)])
    if do_addtimedim:
        var_3d_lst_notime = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (y_dim, x_dim)])
        var_3d_mat = np.array(var_3d_lst + var_3d_lst_notime)
    else:
        var_3d_mat = np.array(var_3d_lst)
    nvar3d = var_3d_mat.size

    var_mat = np.append(var_4d_mat, var_3d_mat)
    nvar = var_mat.size

    
    var_dim = {}
    for vi,var_dat in enumerate(var_4d_mat): var_dim[var_dat] = 4
    for vi,var_dat in enumerate(var_3d_mat): var_dim[var_dat] = 3

    return var_4d_mat, var_3d_mat, var_mat, nvar4d, nvar3d, nvar, var_dim




def load_nc_var_name_list_WW3(tmp_data,sp_dim,t_dim):

    # what are the4d variable names, and how many are there?
    #var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 4])
    #var_4d_mat = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim, z_dim,y_dim, x_dim)])
    #nvar4d = var_4d_mat.size
    #pdb.set_trace()
    #var_3d_mat = np.array([ss for ss in tmp_data.variables.keys() if len(tmp_data.variables[ss].dims) == 3])
    var_mat = np.array([ss for ss in tmp_data.variables.keys() if tmp_data.variables[ss].dims == (t_dim,sp_dim )])
    #pdb.set_trace()
    #nvar3d = var_mat.size

    #var_mat = var_3d_mat
    nvar = var_mat.size


    var_dim = {}
    for vi,var_dat in enumerate(var_mat): var_dim[var_dat] = 3

    return  var_mat,  nvar, var_dim

def rotated_grid_transform(grid_in, option, SP_coor):

    #https://gis.stackexchange.com/questions/10808/manually-transforming-rotated-lat-lon-to-regular-lat-lon/14445#14445


    lon = grid_in[0]
    lat = grid_in[1];

    lon = (lon*np.pi)/180; # Convert degrees to radians
    lat = (lat*np.pi)/180;

    SP_lon = SP_coor[0];
    SP_lat = SP_coor[1];

    theta = 90+SP_lat; # Rotation around y-axis
    phi = SP_lon; # Rotation around z-axis

    theta = (theta*np.pi)/180;
    phi = (phi*np.pi)/180; # Convert degrees to radians

    x = np.cos(lon)*np.cos(lat); # Convert from spherical to cartesian coordinates
    y = np.sin(lon)*np.cos(lat);
    z = np.sin(lat);

    if option == 1: # Regular -> Rotated

        x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z;
        y_new = -np.sin(phi)*x + np.cos(phi)*y;
        z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z;

    else:  # Rotated -> Regular

        phi = -phi;
        theta = -theta;

        x_new = np.cos(theta)*np.cos(phi)*x + np.sin(phi)*y + np.sin(theta)*np.cos(phi)*z;
        y_new = -np.cos(theta)*np.sin(phi)*x + np.cos(phi)*y - np.sin(theta)*np.sin(phi)*z;
        z_new = -np.sin(theta)*x + np.cos(theta)*z;



    lon_new = np.arctan2(y_new,x_new); # Convert cartesian back to spherical coordinates
    lat_new = np.arcsin(z_new);

    lon_new = (lon_new*180)/np.pi; # Convert radians back to degrees
    lat_new = (lat_new*180)/np.pi;

    return lon_new,lat_new


def rotated_grid_from_amm15(lon_in,lat_in, SP_coor = np.array([ -2.5, -37.5])):
    lon_new,lat_new = rotated_grid_transform((lon_in.copy(),lat_in.copy()), 1, SP_coor)
    return lon_new,lat_new

def rotated_grid_to_amm15(lon_in,lat_in, SP_coor = np.array([ -2.5, -37.5])):
    lon_old,lat_old = rotated_grid_transform((lon_in.copy(),lat_in.copy()), 0, SP_coor)
    return lon_old,lat_old


def reduce_rotamm15_grid(nav_lon, nav_lat):
    lon_orig = nav_lon
    lat_orig = nav_lat
    lon_new,lat_new = rotated_grid_from_amm15(lon_orig.copy(),lat_orig.copy(),  SP_coor)

    rot_lon_axis = lon_new.mean(axis = 0)
    rot_lat_axis = lat_new.mean(axis = 1)

    return rot_lon_axis,rot_lat_axis

def testing_rot_pole(nav_lon, nav_lat):
    import matplotlib.pyplot as plt
    lon_orig = nav_lon[::4,::4]
    lat_orig = nav_lat[::4,::4]
    #SP_coor = np.array([177.5-180.,-37.5,])
    lon_new,lat_new = rotated_grid_transform((lon_orig.copy(),lat_orig.copy()), 1, SP_coor)
    lon_old,lat_old = rotated_grid_transform((lon_new.copy(),lat_new.copy()), 0, SP_coor)
    ii,jj = 0,0
    print (RotNPole_lon,RotNPole_lat)
    print (lon_orig[jj,ii],lat_orig[jj,ii])
    print (lon_new[jj,ii]+360,lat_new[jj,ii])
    print (lon_old[jj,ii],lat_old[jj,ii])
    print('Tested with: https://agrimetsoft.com/Cordex%20Coordinate%20Rotation.aspx')
    print('Converted Longitude = 349.11')
    print('Converted Latitude = -7.29')

    lon_new,lat_new = rotated_grid_from_amm15(lon_orig.copy(),lat_orig.copy(),  SP_coor)
    lon_old,lat_old = rotated_grid_to_amm15(lon_new.copy(),lat_new.copy(),  SP_coor)

    plt.figure()
    plt.subplot(2,3,1)
    plt.pcolormesh(lon_orig)
    plt.colorbar()
    plt.contour(lon_orig,colors = 'k')
    plt.subplot(2,3,2)
    plt.pcolormesh(lon_new)
    plt.colorbar()
    plt.contour(lon_new,colors = 'k')
    plt.subplot(2,3,3)
    plt.pcolormesh(lon_old)
    plt.colorbar()
    plt.contour(lon_old,colors = 'k')
    plt.subplot(2,3,4)
    plt.pcolormesh(lat_orig)
    plt.colorbar()
    plt.contour(lat_orig,colors = 'k')
    plt.subplot(2,3,5)
    plt.pcolormesh(lat_new)
    plt.colorbar()
    plt.contour(lat_new,colors = 'k')
    plt.subplot(2,3,6)
    plt.pcolormesh(lat_old)
    plt.colorbar()
    plt.contour(lat_old,colors = 'k')


    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(lon_new.T)
    plt.plot(lon_new.mean(axis = 0),'k', lw = 2)
    plt.subplot(2,2,2)
    plt.plot(lat_new)
    plt.plot(lat_new.mean(axis = 1),'k', lw = 2)
    plt.subplot(2,2,3)
    plt.plot(lon_new.T-lon_new.mean(axis = 0).reshape(365,1))
    plt.subplot(2,2,4)
    plt.plot(lat_new-lat_new.mean(axis = 1).reshape(337,1))
    plt.show()



    pdb.set_trace()

        

def scale_color_map(base_cmap):
    # scales the colourmap with a 4th order polynomial folling ncview, for high, low and linear scales

    defcolmap_lst = [] 
    for i_i in range(256):defcolmap_lst.append(base_cmap(i_i))
    defcolmap_mat = np.array(defcolmap_lst)

    linind = np.linspace(0,1,256)
    polyind = np.linspace(0,1,256)**4
    invpolyind = np.linspace(0,1,256)**(1/4)


    poly_defcolmap_mat = defcolmap_mat.copy()
    invpoly_defcolmap_mat = defcolmap_mat.copy()

    for i_i in range(4):poly_defcolmap_mat[:,i_i] = np.interp(linind, polyind, defcolmap_mat[:,i_i])
    for i_i in range(4):invpoly_defcolmap_mat[:,i_i] = np.interp(linind, invpolyind, defcolmap_mat[:,i_i])

    curr_cmap_low = ListedColormap(poly_defcolmap_mat)
    curr_cmap_high = ListedColormap(invpoly_defcolmap_mat)
    
    return curr_cmap_low,curr_cmap_high



def get_clim_pcolor(ax = None):

    if ax is None:
        ax = plt.gca()
    #    plt.sca(ax)
    #else:
    #    plt.sca(ax)
    # find the 'PolyCollection' associated with the the current ax (i.e. pcolor - assume only has one). find the data assoicated with it. Find the percentile values associated with it. Set the colour clims to these values
    #for child in plt.gca().get_children():
    for child in ax.get_children():
        
        #if child.__class__.__name__ == 'PolyCollection':
        if child.__class__.__name__ in ['PolyCollection','QuadMesh']: # for pcolor and pcolormesh
            clim_out = child.get_clim()
    return clim_out



def set_clim_pcolor(clim_in_tuple,ax = None):
    if ax is None:
        ax = plt.gca()
        #plt.sca(ax)
    #else:
    #    plt.sca(ax)
    #clim_in_tuple = (clim_in_min,clim_in_max)
    # find the 'PolyCollection' associated with the the current ax (i.e. pcolor - assume only has one). find the data assoicated with it. Find the percentile values associated with it. Set the colour clims to these values
    #for child in plt.gca().get_children():
    #print(id(ax), get_clim_pcolor(ax = ax), clim_in_tuple)
    #pdb.set_trace()
    for child in ax.get_children():
        #if child.__class__.__name__ == 'PolyCollection':
        if child.__class__.__name__ in ['PolyCollection','QuadMesh']: # for pcolor and pcolormesh
            child.set_clim(clim_in_tuple)




def set_perc_clim_pcolor(perc_in_min,perc_in_max, sym = False,ax = None):
    if ax is None:
        ax = plt.gca()
        plt.sca(ax)
    #else:
    #    plt.sca(ax)
    perc_in_tuple = (perc_in_min,perc_in_max)
    # find the 'PolyCollection' associated with the the current ax (i.e. pcolor - assume only has one). find the data assoicated with it. Find the percentile values associated with it. Set the colour clims to these values
    for child in ax.get_children():
        #if child.__class__.__name__ == 'PolyCollection':
        if child.__class__.__name__ in ['PolyCollection','QuadMesh']: # for pcolor and pcolormesh
            tmp_data_mat = child.get_array()
            if np.ma.isMA(tmp_data_mat):
                perc_out = np.percentile(tmp_data_mat[tmp_data_mat.mask == False],perc_in_tuple)
            else:
                perc_out = np.percentile(tmp_data_mat,perc_in_tuple)

            if sym == True:
                perc_out = np.abs(perc_out).max()*np.array([-1,1])
            child.set_clim(perc_out)



def set_perc_clim_pcolor_in_region(perc_in_min,perc_in_max, illtype = 'pcolor', perc = True, set_not_get = True,ax = None,sym = False):

    if ax is None:
        ax = plt.gca()
        plt.sca(ax)
    else:
        plt.sca(ax)


    perc_in_tuple = (perc_in_min,perc_in_max)
    # find the 'PolyCollection' associated with the the current ax (i.e. pcolor - assume only has one). find the data assoicated with it. Find the percentile values associated with it. Set the colour clims to these values
    xlim = np.array(plt.gca().get_xlim()).copy()
    ylim = np.array(plt.gca().get_ylim()).copy()
    # incase e.g. ax.invert_yaxis()
    xlim.sort()
    ylim.sort()

    if illtype.lower() in ['pcolor','pcolormesh']:

        for child in plt.gca().get_children():
            #if child.__class__.__name__ == 'PolyCollection':
            if child.__class__.__name__ in ['PolyCollection','QuadMesh']: # for pcolor and pcolormesh


                if perc:

                    tmp_data_mat = child.get_array()

                    #xylocs = plt.gca().get_children()[0]._coordinates

                    xylocs = child._coordinates
                    xs = xylocs[:-1,:-1,0].ravel()
                    ys = xylocs[:-1,:-1,1].ravel()

                    tmpdata_in_screen_ind = (xs>xlim[0]) &  (xs<xlim[1]) & (ys>ylim[0]) &  (ys<ylim[1])
                    tmpdata_in_screen = tmp_data_mat.ravel()[tmpdata_in_screen_ind]

                    if np.ma.isMA(tmpdata_in_screen):
                        tmp_data_in_screen = tmpdata_in_screen[tmpdata_in_screen.mask == False]
                        if len(tmp_data_in_screen) == 0: 
                            return
                            pdb.set_trace()
                        perc_out = np.percentile(tmp_data_in_screen,perc_in_tuple)
                    else:
                        perc_out = np.percentile(tmpdata_in_screen,perc_in_tuple)

                else:
                    perc_out = (perc_in_min,perc_in_max)


                if sym == True:
                    perc_out = np.abs(perc_out).max()*np.array([-1,1])

                if set_not_get:

                    child.set_clim(perc_out)

                else:
                    return perc_out

    elif (illtype.lower() == 'scatter'):
        for child in plt.gca().get_children():
            if child.__class__.__name__ == 'PathCollection':




                if perc:


                    tmp_data_mat = child.get_array()


                    xs = child.get_offsets()[:,0]
                    ys = child.get_offsets()[:,1]

                    tmpdata_in_screen_ind = (xs>xlim[0]) &  (xs<xlim[1]) & (ys>ylim[0]) &  (ys<ylim[1])
                    tmpdata_in_screen = tmp_data_mat[tmpdata_in_screen_ind]


                    if np.ma.isMA(tmpdata_in_screen):
                        perc_out = np.percentile(tmpdata_in_screen[tmpdata_in_screen.mask == False],perc_in_tuple)
                    else:
                        perc_out = np.percentile(tmpdata_in_screen,perc_in_tuple)



                else:
                    perc_out = (perc_in_min,perc_in_max)


                if sym == True:
                    perc_out = np.abs(perc_out).max()*np.array([-1,1])

                if set_not_get:

                    child.set_clim(perc_out)

                else:
                    return perc_out




def get_colorbar_values(cb, verbose = False):



    #print ('Think this is simpler with Python3')
    return cb.ax.get_yticks()


def get_pnts_pcolor_in_region(illtype = 'pcolor', ax = None):

    if ax is None:
        ax = plt.gca()
        plt.sca(ax)
    else:
        plt.sca(ax)


    # find the 'PolyCollection' associated with the the current ax (i.e. pcolor - assume only has one). find the data assoicated with it. Find the percentile values associated with it. Set the colour clims to these values
    xlim = np.array(plt.gca().get_xlim()).copy()
    ylim = np.array(plt.gca().get_ylim()).copy()
    # incase e.g. ax.invert_yaxis()
    xlim.sort()
    ylim.sort()
    npnts = np.nan
    #pdb.set_trace()
    if illtype.lower() in ['pcolor','pcolormesh']:

        for child in plt.gca().get_children():
            #if child.__class__.__name__ == 'PolyCollection':
            if child.__class__.__name__ in ['PolyCollection','QuadMesh']: # for pcolor and pcolormesh

                    tmp_data_mat = child.get_array()

                    #xylocs = plt.gca().get_children()[0]._coordinates

                    xylocs = child._coordinates
                    xs = xylocs[:-1,:-1,0].ravel()
                    ys = xylocs[:-1,:-1,1].ravel()

                    tmpdata_in_screen_ind = (xs>xlim[0]) &  (xs<xlim[1]) & (ys>ylim[0]) &  (ys<ylim[1])
                    tmpdata_in_screen = tmp_data_mat.ravel()[tmpdata_in_screen_ind]
                    npnts = tmpdata_in_screen.size

    elif (illtype.lower() == 'scatter'):
        for child in plt.gca().get_children():
            if child.__class__.__name__ == 'PathCollection':

                    tmp_data_mat = child.get_array()

                    xs = child.get_offsets()[:,0]
                    ys = child.get_offsets()[:,1]

                    tmpdata_in_screen_ind = (xs>xlim[0]) &  (xs<xlim[1]) & (ys>ylim[0]) &  (ys<ylim[1])
                    tmpdata_in_screen = tmp_data_mat[tmpdata_in_screen_ind]
                    npnts = tmpdata_in_screen.size

    #pdb.set_trace()
    return npnts








def field_gradient_2d(tmpdat_in,e1t_in,e2t_in,
                      do_mask = False,curr_griddict= None,
                      meth_2d=0,meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False):

    # Copy inputs, so can't affect originals.
    e1t = e1t_in.copy()
    e2t = e2t_in.copy()
    
    if abs_pre:
        tmpdat = np.abs(tmpdat_in.copy())
    else:
        tmpdat = tmpdat_in.copy()

    # Where land suppression is used, can have bad values in land processors (i.e. 1e308).
    # if negative cell width/height, set to zero. if greater then 1000km, set to zero
    e1t[e1t<0] = 0
    e1t[e1t>1e6] = 0
    e2t[e2t<0] = 0
    e2t[e2t>1e6] = 0

    # if masking land points, set land cell width/height to zero, and land data points to np.ma.masked
    if do_mask:
        tmpmask = curr_griddict['tmask'][0] == False

        e1t[tmpmask] = 0
        e1t[tmpmask] = 0
        e2t[tmpmask] = 0
        e2t[tmpmask] = 0
        tmpdat = np.ma.array(tmpdat, mask  = tmpmask )



    nlat,nlon = e1t.shape


    xs = e1t.cumsum(axis = 1)
    ys = e2t.cumsum(axis = 0)

    if meth == 0: # centred difference:

        if dx_d_dx:
            dtmpdat_dx_c = (tmpdat[1:-1,2:] - tmpdat[1:-1,:-2])
            dtmpdat_dy_c = (tmpdat[2:,1:-1] - tmpdat[:-2,1:-1])
        else:
            dtmpdat_dx_c = (tmpdat[1:-1,2:] - tmpdat[1:-1,:-2])/(0.001*2*(xs[1:-1,2:] - xs[1:-1,:-2]))
            dtmpdat_dy_c = (tmpdat[2:,1:-1] - tmpdat[:-2,1:-1])/(0.001*2*(ys[2:,1:-1] - ys[:-2,1:-1]))

        dtmpdat_dkm = np.sqrt( (dtmpdat_dx_c)**2 + (dtmpdat_dy_c)**2   )


        dtmpdat_dkm_out = np.ma.zeros((nlat,nlon))
        dtmpdat_dkm_out[:] = np.ma.masked
        if   meth_2d == 0:
            dtmpdat_dkm_out[1:-1,1:-1] = dtmpdat_dkm
        elif meth_2d == 1:
            dtmpdat_dkm_out[1:-1,1:-1] = dtmpdat_dx_c
        elif meth_2d == 2:
            dtmpdat_dkm_out[1:-1,1:-1] = dtmpdat_dy_c

    elif meth == 1:

        if dx_d_dx:
            dtmpdat_dx_c = (tmpdat[:-1,1:] - tmpdat[:-1,:-1])
            dtmpdat_dy_c = (tmpdat[1:,:-1] - tmpdat[:-1,:-1])
        else:
            dtmpdat_dx_c = (tmpdat[:-1,1:] - tmpdat[:-1,:-1])/(0.001*2*(xs[:-1,1:] - xs[:-1,:-1]))
            dtmpdat_dy_c = (tmpdat[1:,:-1] - tmpdat[:-1,:-1])/(0.001*2*(ys[1:,:-1] - ys[:-1,:-1]))

        dtmpdat_dkm = np.sqrt( (dtmpdat_dx_c)**2 + (dtmpdat_dy_c)**2   )


        dtmpdat_dkm_out = np.ma.zeros((nlat,nlon))
        dtmpdat_dkm_out[:] = np.ma.masked

        if   meth_2d == 0:
            dtmpdat_dkm_out[:-1,:-1] = dtmpdat_dkm
        elif meth_2d == 1:
            dtmpdat_dkm_out[:-1,:-1] = dtmpdat_dx_c
        elif meth_2d == 2:
            dtmpdat_dkm_out[:-1,:-1] = dtmpdat_dy_c

    if abs_post:
        dtmpdat_dkm_out = np.abs(dtmpdat_dkm_out)
    
    return dtmpdat_dkm_out




def interp1dmat_wgt(indata, wgt_tuple):
    # ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask = interp1dmat_create_weight(gdept,z_lev,use_xarray_gdept = False)
    # outdata = interp1dmat_wgt(indata, ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask)
    # wgt_tuple = interp1dmat_create_weight(gdept,z_lev,use_xarray_gdept = False)
    # outdata = interp1dmat_wgt(indata, wgt_tuple)
    
    
    ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask = wgt_tuple
    if len(indata.shape) == 3:
        outdata = np.ma.array(wgt1*indata[ind1,xind_mat,yind_mat] + wgt2*indata[ind2,xind_mat,yind_mat])
        outdata.mask = outdata.mask | wgt_mask
    elif len(indata.shape) == 4:
        nt = indata.shape[0]
        outdata = np.ma.zeros((nt,) + indata.shape[2:])*np.ma.masked
        for ti in range(nt): outdata[ti,:,:] = np.ma.array(wgt1*indata[ti,ind1,xind_mat,yind_mat] + wgt2*indata[ti,ind2,xind_mat,yind_mat])
        for ti in range(nt): outdata[ti,:,:].mask = outdata[ti,:,:].mask | wgt_mask
        #pdb.set_trace()
    return outdata

def interp1dmat_create_weight(gdept,z_lev,use_xarray_gdept = False):
    # ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask = interp1dmat_create_weight(gdept,z_lev,use_xarray_gdept = False)

    import socket
    computername = socket.gethostname()
    comp = 'linux'
    if computername in ['xcel00','xcfl00']: comp = 'hpc'

    ## Trouble shooting HPC seg fault
    verbose_debugging = False
    if verbose_debugging: from datetime import datetime
    if verbose_debugging: import pdb
    nz,nlon,nlat = gdept.shape


    

    
    if verbose_debugging: print('gdept_ma', datetime.now())
    if use_xarray_gdept:
        gdept_ma = np.array(gdept)
    else:
        gdept_ma = gdept
    gdept_ma_min = gdept_ma.min(axis = 0)
    gdept_ma_max = gdept_ma.max(axis = 0)
    #gdept_ma_ptp = gdept_ma.ptp(axis = 0)
    gdept_ma_ptp = np.ptp(gdept_ma,axis = 0)

    if verbose_debugging: print('x_mat, y_mat', datetime.now())

    if use_xarray_gdept:
        xind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
        yind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
    else:
        xind_mat = np.zeros(gdept_ma.shape[1:], dtype = 'int')
        yind_mat = np.zeros(gdept_ma.shape[1:], dtype = 'int')
    #for zi in range(nz): zind_mat[zi,:,:] = zi
    for xi in range(nlon): xind_mat[xi,:] = xi
    for yi in range(nlat): yind_mat[:,yi] = yi


    if use_xarray_gdept:
        if verbose_debugging: print('ind1', datetime.now())
        ind1 = (gdept_ma<z_lev).sum(axis = 0).astype('int')
        ind1[ind1 == nz] = 0
        if verbose_debugging: print('ind2', datetime.now())
        ind2 = (nz-1)-(gdept_ma>z_lev).sum(axis = 0).astype('int')

    else:
        if verbose_debugging: print('ind1', datetime.now())
        ind1 = (gdept_ma<z_lev).sum(axis = 0).data.astype('int')
        ind1[ind1 == nz] = 0
        if verbose_debugging: print('ind2', datetime.now())
        ind2 = (nz-1)-(gdept_ma>z_lev).sum(axis = 0).data.astype('int')
    #tmpind2 = (gdept_ma>z_lev).sum(axis = 0).data.astype('int')
    #ind2 = (nz-1)-tmpind2

    #plt.pcolormesh(gdept_ma[ind1,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()
    #plt.pcolormesh(gdept_ma[ind2,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()

    if verbose_debugging: print('z_ind1', datetime.now())
    z_ind1 = gdept_ma[ind1,xind_mat,yind_mat]
    if verbose_debugging: print('z_ind2', datetime.now())
    z_ind2 = gdept_ma[ind2,xind_mat,yind_mat]
    dz_ind = z_ind1-z_ind2

    if verbose_debugging: print('zdist', datetime.now())
    zdist1 = z_ind1 - z_lev
    zdist2 = z_lev - z_ind2

    if verbose_debugging: print('zdist_norm', datetime.now())
    if comp == 'hpc':
        zdist1_norm = zdist1.copy()
        zdist2_norm = zdist1.copy()

        for xi in range(nlon):
            for yi in range(nlat): zdist1_norm[xi,yi] = zdist1[xi,yi]/dz_ind[xi,yi]
            for yi in range(nlat): zdist2_norm[xi,yi] = zdist2[xi,yi]/dz_ind[xi,yi]
            
    else:
        zdist1_norm = zdist1/dz_ind
        zdist2_norm = zdist2/dz_ind


    if verbose_debugging: print('wgt', datetime.now())
    wgt1 = zdist2_norm
    wgt2 = zdist1_norm

    if verbose_debugging: print('wgt_mask', datetime.now())

    wgt_mask = (z_lev > gdept_ma_max) | (z_lev < gdept_ma_min) | (gdept_ma_ptp<1)

    if z_lev == 0:
        ind1[:,:]= 1
        ind2[:,:]= 0
        wgt1[:,:]= 0.
        wgt2[:,:]= 1.
        #wgt_mask = gdept_ma[0] == 0.1
        wgt_mask[:] = False

    return ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask

    pdb.set_trace()




def lon_lat_to_str(lon,lat,lonlatstr_format = '%.2f'):
    
    degree_sign= u'\N{DEGREE SIGN}'
    #pdb.set_trace()
    
    if lat>=0:
        latstr = (lonlatstr_format+'%sN')%(abs(lat),degree_sign)
        latstrfname = (lonlatstr_format+'N')%(abs(lat))
    else:
        latstr = (lonlatstr_format+'%sS')%(abs(lat),degree_sign)
        latstrfname = (lonlatstr_format+'S')%(abs(lat))
    
    if lon>=0:
        lonstr = (lonlatstr_format+'%sE')%(abs(lon),degree_sign)
        lonstrfname = (lonlatstr_format+'E')%(abs(lon))
    else:
        lonstr = (lonlatstr_format+'%sW')%(abs(lon),degree_sign)
        lonstrfname = (lonlatstr_format+'W')%(abs(lon))

    lat_lon_str = '%s %s'%(latstr, lonstr)
    lat_lon_strfname = '%s_%s'%(latstrfname, lonstrfname)

    return lat_lon_str,lonstr,latstr,lat_lon_strfname




def ismask(tmpvar):

    ismask_out = False
    if isinstance(tmpvar,np.ma.core.MaskedArray):
        ismask_out = True

    return ismask_out


def nearbed_int_index_val(tmp3dmasknbivar):
    nbzindint,nbiindint,nbijndint,tmask = nearbed_int_index_func(tmp3dmasknbivar)

    tmp_nb_mat = nearbed_int_use_index_val(tmp3dmasknbivar,nbzindint,nbiindint,nbijndint,tmask)
    
    return tmp_nb_mat

def nearbed_int_use_index_val(tmp3dmasknbivar,nbzindint,nbiindint,nbijndint,tmask):
    
    tmp_nb_mat = np.ma.masked_invalid(tmp3dmasknbivar[nbzindint,nbiindint,nbijndint])
    
    return tmp_nb_mat


def nearbed_int_index_func(tmp3dmasknbivar):

    tmask = tmp3dmasknbivar.mask

    nz, nj, ni = tmask.shape

    nbzindint = np.maximum(((tmask == False).sum(axis = 0)-1),0).astype('int')
    nbiindint,nbijndint = np.meshgrid(np.arange(nj).astype('int'),np.arange(ni).astype('int'))

    #tmpnbt = tmp3dmasknbivar[nbindint.astype('int'),nbiindint.astype('int').T,nbijndint.astype('int').T]

    nbiindint,nbijndint = nbiindint.astype('int').T,nbijndint.astype('int').T


    return nbzindint,nbiindint,nbijndint,tmask


def nearbed_index_func(tmp_var):

    if ismask(tmp_var):
        tmask = tmp_var.mask.copy()
    else:
        tmp_var = np.ma.masked_equal(tmp_var, 0)
        tmask = tmp_var.mask.copy()

    nz, nj, ni = tmask.shape


    # make an array of the size of the domain with the level numbers
    #zindmat = np.tile(np.arange(51),(297,375,1)).T
    zindmat = np.tile(np.arange(nz),(ni, nj,1)).T

    # Multiply this by the domain mask, so grid boxes below the sea bed are set to zero
    zindmatT = zindmat*(~tmask)

    # Identify the maximum model level for each grid box.
    zindmax = zindmatT.max(axis = 0)

    # ... and turn this into a 3d array, matching the domain grid size
    #zindmaxmat = np.tile(zindmax,(51,1,1))
    zindmaxmat = np.tile(zindmax,(nz,1,1))

    # Find where the grid boxes are not the near bed values (so True where we want to mask)
    nbind = zindmaxmat != zindmat

    #if ((nbind*1).sum(axis = 0).min() != 50) | ((nbind*1).sum(axis = 0).max() != 50) :
    #if ((nbind.astype('int')).sum(axis = 0).min() != (nz-1)) | ((nbind.astype('int')).sum(axis = 0).max() != (nz-1)) :
    if ((nbind).sum(axis = 0).min() != (nz-1)) | ((nbind).sum(axis = 0).max() != (nz-1)) :
        print("ERROR, nbind has found more than one near bed boxes...")
        pdb.set_trace()


    return nbind,tmask



def weighted_depth_mean_masked_var(tmpvar_in, e3_in,output_weighting = False):
    # tmpvar_in and e3_in must be 3d (nz, ny, nx)
    
    #mask e3 with 3d variable mask
    e3_ma = np.ma.array(e3_in,mask = tmpvar_in.mask)
    e3_ma_sum = e3_ma.sum(axis = 0)

    dz_wgt = e3_ma/e3_ma_sum

    DM_out = (tmpvar_in*dz_wgt).sum(axis = 0)

    
    if output_weighting:
        return DM_out,dz_wgt
    else:
        return DM_out



def interp_UV_vel_to_Tgrid(tmp_DMU_in,tmp_DMV_in):
    numdim = len(tmp_DMU_in.shape)
    if numdim == 2:
        tmp_DMU = tmp_DMU_in
        tmp_DMV = tmp_DMV_in
    else:
        print('Can only have 2 dimension')
        pdb.set_trace()

    nlat_int, nlon_int = tmp_DMU.shape
    DMU_T = np.ma.zeros((nlat_int, nlon_int)) * np.ma.masked
    DMV_T = np.ma.zeros((nlat_int, nlon_int)) * np.ma.masked
    DMUV_T = np.ma.zeros((nlat_int, nlon_int)) * np.ma.masked

    # Interpolate DMU and DMV from U and V grid, to T grid.
    # Confirmed with Jeff Polton, National Oceanography Centre
    DMU_T[ 1:, 1:] = (
        tmp_DMU[ 1:, :-1] + tmp_DMU[ 1:, 1:]
    ) / 2.0
    DMV_T[ 1:, 1:] = (
        tmp_DMV[ :-1, 1:] + tmp_DMV[ 1:, 1:]
    ) / 2.0

    # Calculate the barotropic current speed
    DMUV_T = np.sqrt(DMU_T**2 + DMV_T**2)

    return DMU_T, DMV_T, DMUV_T



def current_barb(x1,y1,u_in,v_in,evx = 2,evy = 2,x0 = 0,y0 = 2,arrow_style = 'barb',fixed_len = 0.05,scf = 4,cutoff_perc = [10., 20., 30., 40., 50., 60., 70., 80., 90.], cutoff = None, no_zero_vel = True,ax = None,**kwargs):

    '''
    evx = 2
    evy = 2
    x0 = 0
    y0 = 2
    arrow_style = 'barb'
    fixed_len = 0.05
    scf = 4
    cutoff_perc = [10., 20., 30., 40., 50., 60., 70., 80., 90.]
    cutoff = None
    no_zero_vel = True
    ax = None
    '''

    if ax is not None: plt.sca(ax)


    u = np.ma.array(u_in.copy())
    v = np.ma.array(v_in.copy())

    if (x1.shape == y1.shape == u.shape == v.shape) == False:
        print('not all inputs the same shape')
        return None
    #x1,y1,u,v = nav_lon,nav_lat, nt_dmu_djf,nt_dmv_djf

    if no_zero_vel:
        ma = (np.ma.masked_equal(u,0.)*np.ma.masked_equal(v,0.)).mask
    else:
        ma = (u*v).mask

    u.mask = ma
    v.mask = ma
    u.data[u.mask] =0.
    v.data[v.mask] =0.

    uv = np.sqrt(u**2 + v**2)

    if fixed_len is None:
        uvlen = uv.copy()
    else:
        uvlen = uv.copy()*0. + fixed_len


    if uv.mask.all():
        print('UV fully masked')
        return None

    ang = np.arctan2(v,u)
    ang2 = ang + np.pi/2.

    dx = uvlen * np.cos(ang)*scf
    dy = uvlen * np.sin(ang)*scf
    dx2 = uvlen * np.cos(ang2)*scf
    dy2 = uvlen * np.sin(ang2)*scf
    x2 = x1+dx
    y2 = y1+dy

    uvlenr = (uvlen)[y0::evy,x0::evx].ravel().reshape(1,-1)
    uvr = (uv)[y0::evy,x0::evx].ravel().reshape(1,-1)
    x1r = (x1)[y0::evy,x0::evx].ravel().reshape(1,-1)
    x2r = (x2)[y0::evy,x0::evx].ravel().reshape(1,-1)
    dxr = (dx)[y0::evy,x0::evx].ravel().reshape(1,-1)
    dx2r = (dx2)[y0::evy,x0::evx].ravel().reshape(1,-1)
    y1r = (y1)[y0::evy,x0::evx].ravel().reshape(1,-1)
    y2r = (y2)[y0::evy,x0::evx].ravel().reshape(1,-1)
    dyr = (dy)[y0::evy,x0::evx].ravel().reshape(1,-1)
    dy2r = (dy2)[y0::evy,x0::evx].ravel().reshape(1,-1)


    hw = 0.15
    hl = 0.25
    hl1 = 1.-hl
    bsp = 0.15


    #lines
    if arrow_style.lower() == 'lines':
        x12 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr)    ),axis =0).T.ravel()
        y12 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr)    ),axis =0).T.ravel()

    if arrow_style.lower() == 'flick':
        x12 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r)    ),axis =0).T.ravel()
        y12 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r)    ),axis =0).T.ravel()

    #triangles
    if arrow_style.lower() == 'triangles':
        x12 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr*hl1),(x1r+dxr*hl1+hw*dx2r),(x1r+dxr),(x1r+dxr*hl1-hw*dx2r),( x1r+dxr*hl1)    ),axis =0).T.ravel()
        y12 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr*hl1),(y1r+dyr*hl1+hw*dy2r),(y1r+dyr),(y1r+dyr*hl1-hw*dy2r),( y1r+dyr*hl1)    ),axis =0).T.ravel()
    #hollow triangles
    elif arrow_style.lower() == 'hollowtriangles':
        x12 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r),(x1r+dxr*hl1-hw*dx2r),(x1r+dxr)   ),axis =0).T.ravel()
        y12 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r),(y1r+dyr*hl1-hw*dy2r),(y1r+dyr)   ),axis =0).T.ravel()
    #arrows
    elif arrow_style.lower() == 'arrows':
        x12 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r),(x1r+dxr),(x1r+dxr*hl1-hw*dx2r),    ),axis =0).T.ravel()
        y12 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r),(y1r+dyr),(y1r+dyr*hl1-hw*dy2r),    ),axis =0).T.ravel()

    elif arrow_style.lower() == 'barb':
        # barbs
        x12_barb_00 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr)   ),axis =0).T
        y12_barb_00 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr)   ),axis =0).T
        x12_barb_10 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r), (x1r+dxr)   ),axis =0).T
        y12_barb_10 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r), (y1r+dyr)   ),axis =0).T
        x12_barb_20 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*(hl1-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr)   ),axis =0).T
        y12_barb_20 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*(hl1-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr)   ),axis =0).T
        x12_barb_30 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*(hl1-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr), (x1r+((1-2*bsp)*dxr)),(x1r+(dxr*(hl1-2*bsp))+hw*dx2r), (x1r+((1-2*bsp)*dxr))   ),axis =0).T
        y12_barb_30 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*(hl1-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr), (y1r+((1-2*bsp)*dyr)),(y1r+(dyr*(hl1-2*bsp))+hw*dy2r), (y1r+((1-2*bsp)*dyr))   ),axis =0).T
        x12_barb_40 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*hl1+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*(hl1-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr), (x1r+((1-2*bsp)*dxr)),(x1r+(dxr*(hl1-2*bsp))+hw*dx2r), (x1r+((1-2*bsp)*dxr)), (x1r+((1-3*bsp)*dxr)),(x1r+(dxr*(hl1-3*bsp))+hw*dx2r), (x1r+((1-3*bsp)*dxr))   ),axis =0).T
        y12_barb_40 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*hl1+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*(hl1-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr), (y1r+((1-2*bsp)*dyr)),(y1r+(dyr*(hl1-2*bsp))+hw*dy2r), (y1r+((1-2*bsp)*dyr)), (y1r+((1-3*bsp)*dyr)),(y1r+(dyr*(hl1-3*bsp))+hw*dy2r), (y1r+((1-3*bsp)*dyr))   ),axis =0).T

        x12_barb_05 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*(1. - hl/2)+hw*dx2r/2), (x1r+dxr)   ),axis =0).T
        y12_barb_05 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*(1. - hl/2)+hw*dy2r/2), (y1r+dyr)   ),axis =0).T
        x12_barb_15 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*(1. - hl)+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*((1. - hl/2)-bsp))+hw*dx2r/2), (x1r+(1-bsp)*dxr)   ),axis =0).T
        y12_barb_15 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*(1. - hl)+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*((1. - hl/2)-bsp))+hw*dy2r/2), (y1r+(1-bsp)*dyr)   ),axis =0).T
        x12_barb_25 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*(1. - hl)+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*((1. - hl)-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr), (x1r+((1-2*bsp)*dxr)),(x1r+(dxr*((1. - hl/2)-2*bsp))+hw*dx2r/2), (x1r+((1-2*bsp)*dxr))   ),axis =0).T
        y12_barb_25 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*(1. - hl)+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*((1. - hl)-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr), (y1r+((1-2*bsp)*dyr)),(y1r+(dyr*((1. - hl/2)-2*bsp))+hw*dy2r/2), (y1r+((1-2*bsp)*dyr))   ),axis =0).T
        x12_barb_35 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*(1. - hl)+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*((1. - hl)-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr), (x1r+((1-2*bsp)*dxr)),(x1r+(dxr*((1. - hl)-2*bsp))+hw*dx2r), (x1r+((1-2*bsp)*dxr)), (x1r+((1-3*bsp)*dxr)),(x1r+(dxr*((1. - hl/2)-3*bsp))+hw*dx2r/2), (x1r+((1-3*bsp)*dxr))   ),axis =0).T
        y12_barb_35 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*(1. - hl)+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*((1. - hl)-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr), (y1r+((1-2*bsp)*dyr)),(y1r+(dyr*((1. - hl)-2*bsp))+hw*dy2r), (y1r+((1-2*bsp)*dyr)), (y1r+((1-3*bsp)*dyr)),(y1r+(dyr*((1. - hl/2)-3*bsp))+hw*dy2r/2), (y1r+((1-3*bsp)*dyr))   ),axis =0).T

        x12_barb_45 = np.concatenate( (  x1r*np.nan,x1r,(x1r+dxr),(x1r+dxr*(1. - hl)+hw*dx2r), (x1r+dxr), (x1r+(1-bsp)*dxr),(x1r+(dxr*((1. - hl)-bsp))+hw*dx2r), (x1r+(1-bsp)*dxr), (x1r+((1-2*bsp)*dxr)),(x1r+(dxr*((1. - hl)-2*bsp))+hw*dx2r), (x1r+((1-2*bsp)*dxr)), (x1r+((1-3*bsp)*dxr)),(x1r+(dxr*((1. - hl)-3*bsp))+hw*dx2r), (x1r+((1-3*bsp)*dxr)), (x1r+((1-4*bsp)*dxr)),(x1r+(dxr*((1. - hl/2)-4*bsp))+hw*dx2r/2), (x1r+((1-4*bsp)*dxr))   ),axis =0).T
        y12_barb_45 = np.concatenate( (  y1r*np.nan,y1r,(y1r+dyr),(y1r+dyr*(1. - hl)+hw*dy2r), (y1r+dyr), (y1r+(1-bsp)*dyr),(y1r+(dyr*((1. - hl)-bsp))+hw*dy2r), (y1r+(1-bsp)*dyr), (y1r+((1-2*bsp)*dyr)),(y1r+(dyr*((1. - hl)-2*bsp))+hw*dy2r), (y1r+((1-2*bsp)*dyr)), (y1r+((1-3*bsp)*dyr)),(y1r+(dyr*((1. - hl)-3*bsp))+hw*dy2r), (y1r+((1-3*bsp)*dyr)), (y1r+((1-4*bsp)*dyr)),(y1r+(dyr*((1. - hl/2)-4*bsp))+hw*dy2r/2), (y1r+((1-4*bsp)*dyr))   ),axis =0).T

        #cutoff_perc = 100.*np.arange(1.,10.)/10.

        if cutoff is None:
            cutoff=np.percentile(uvr[uvr.mask==False], cutoff_perc)

        ind_lst = []
        ind_lst.append(uvr< cutoff[1])
        for ii in range(0,cutoff.size-1): ind_lst.append((uvr >= cutoff[ii]) & (uvr< cutoff[ii+1]))
        ind_lst.append((uvr >= cutoff[-1]))


        x12_lst = []
        y12_lst = []
        for tmpind,xbarb in zip(ind_lst,[x12_barb_00,x12_barb_05,x12_barb_10,x12_barb_15,x12_barb_20,x12_barb_25,x12_barb_30,x12_barb_35,x12_barb_40,x12_barb_45]):x12_lst.append(xbarb[tmpind.ravel(),:].ravel())
        for tmpind,ybarb in zip(ind_lst,[y12_barb_00,y12_barb_05,y12_barb_10,y12_barb_15,y12_barb_20,y12_barb_25,y12_barb_30,y12_barb_35,y12_barb_40,y12_barb_45]):y12_lst.append(ybarb[tmpind.ravel(),:].ravel())
        x12,y12 = [jj for ii in x12_lst for jj in ii ],[jj for ii in y12_lst for jj in ii ]


    #plt.plot(np.ma.array(x1, mask = ma)[::evy,::evx].ravel(),y1[::evy,::evx].ravel(),'ko',ms = 2.5)
    handle = plt.plot(x12,y12,**kwargs)
    #plt.axis('square')
    return handle







def mask_stats(data,mask,sparse=False):
    # MLD in mixed areas are masked out, so doesn't include them in the stats. Therefore MLD the stats are only for stratified regions.
    #  not the case with PEA as unstratified areas have a PEA of 0 J/m3  e.g. # sttmpdata.mask = (sttmpdata.data < 10) | sttmpdata.mask


    #tmpdata.mean() # mean as used
    #tmpdata[tmpdata.mask == False].sum()/tmpdata[tmpdata.mask == False].size # mean from the sum of the un masked areas and number of unmasked gridboxes
    #tmpdata.sum()/tmpdata.size # mean from the sum of the all areas and number of all gridboxes

    # if using sparse data (i.e. EN4) use sparse = True - this stops the mask-data check, and the region = 0 "out" stat, which has memory errors.

    reg = np.unique(mask)

    nreg = reg.size

    tot = np.zeros(nreg)
    tot_sq = np.zeros(nreg)
    ave = np.zeros(nreg)
    var = np.zeros(nreg)
    std = np.zeros(nreg)
    cnt = np.zeros(nreg)
    out = np.zeros(nreg)
    min = np.zeros(nreg)
    max = np.zeros(nreg)
    ssq = np.zeros(nreg)


    for regi,regind in enumerate(reg):
        if regind == 0:
            tmpind = mask!=regind
        else:
            tmpind = mask==regind

        if sparse == False:
            if data[tmpind].mask.any():
                print("stop as mask-data disagreement")
                pdb.set_trace()
        tmpdata = data[tmpind]
        #if regi == 1: pdb.set_trace()


        if sparse == False: # if
            cnt[regi] = tmpdata.size
        else:
            cnt[regi] = (tmpdata.mask == False).sum()
        tot[regi] = tmpdata.sum()
        tot_sq[regi] = (tmpdata**2).sum()
        ave[regi] = tmpdata.mean()
        var[regi] = tmpdata.var()
        min[regi] = tmpdata.min()
        max[regi] = tmpdata.max()
        std[regi] = np.sqrt(tmpdata.var())
        #pdb.set_trace()
        if sparse == False: # if
            out[regi] = np.outer(tmpdata,tmpdata).sum() / (cnt[regi]**2)
        else:
            out[regi] = np.nan
        ssq[regi] = (tmpdata**2).mean()




    stats = {}
    stats['cnt'] = cnt
    stats['ave'] = ave
    stats['tot'] = tot
    stats['tot_sq'] = tot_sq
    stats['var'] = var
    stats['std'] = std
    stats['out'] = out
    stats['ssq'] = ssq
    stats['min'] = min
    stats['max'] = max





    spat_var = ave**2*(cnt - 1) - tot**2/cnt  + tot_sq/cnt



    stats['spat_var'] = spat_var

    return ave,stats




def sw_dens(tdata,sdata):

    #outputs rho density, not sigma (density - 1000)
    #function sw_dens,tdata,sdata,unesco=unesco
    #; Consistent with the ukmo routine calc_dens
    #; tdata = potential temperature i.e. stash 101
    #; sdata = salinity, not stash 102 (stash 102*1000 + 35)

 
    TO=13.4992332
    SO=-0.0022500
    SIGO=24.573651
    #    C=[-.2017283E-03,0.7710054E+00,-.4918879E-05,-.2008622E-02,$
    #      0.4495550E+00,0.3656148E-07,0.4729278E-02,0.3770145E-04,$
    #      0.6552727E+01]
    C=[-.2017283E-03,0.7710054E+00,-.4918879E-05,-.2008622E-02,0.4495550E+00,0.3656148E-07,0.4729278E-02,0.3770145E-04,0.6552727E+01]


#    unesco = 0
#    #  ;
#    #  ;  UNESCO values
#    #  ;
#    if unesco == 1:
#        TO=13.4993292
#        SO=-0.0022500
#        SIGO=24.573975
#        #    C=[-0.2016022E-03,0.7730564E+00,-0.4919295E-05,-0.2022079E-02,$
#        #      0.3168986E+00,0.3610338E-07, 0.3777372E-02, 0.3603786E-04,$
#        #      0.1609520E+01]
#        C=[-0.2016022E-03,0.7730564E+00,-0.4919295E-05,-0.2022079E-02,0.3168986E+00,0.3610338E-07, 0.3777372E-02, 0.3603786E-04,0.1609520E+01]


    TQ=tdata-TO
    #;            SQ=(salt.data(index)-35.0)/1000.-SO
    SQ=(sdata-35.)/1000-SO
    #SQ=sdata-SO

    #if is_cube :
    #dens_out =  ( C[0] + (C[3] + C[6]*SQ)*SQ+(C[2] + C[7]*SQ + C[5]*TQ )*TQ ) * TQ+( C[1] + (C[4] + C[8]*SQ)*SQ ) * SQ * 1000. + SIGO
    #dens_out =  (( C[0] + (C[3] + C[6]*SQ)*SQ+(C[2] + C[7]*SQ + C[5]*TQ )*TQ )* TQ   +  ( C[1] + (C[4] + C[8]*SQ)*SQ )   )     * SQ * 1000. + SIGO

    #dens_out =  ( C[0] + (C[3] + C[6]*SQ.data)*SQ.data+(C[2] + C[7]*SQ.data + C[5]*TQ.data )*TQ.data ) * TQ.data+( C[1] + (C[4] + C[8]*SQ.data)*SQ.data ) * SQ.data * 1000. + SIGO


    part1 = (C[0] + (C[3] + C[6]*SQ)*SQ + (C[2] + C[7]*SQ + C[5]*TQ)*TQ)*TQ
    #if is_cube :
    #
    part2 = ( C[1] + (C[4] + C[8]*SQ)*SQ)*SQ
    #if is_cube :
    sigma_dens_out =  (part1  +  part2)   * 1000. + SIGO
    #dens_out =  ( C[0] + (C[3] + C[6]*SQ.data)*SQ.data+(C[2] + C[7]*SQ.data + C[5]*TQ.data )*TQ.data ) * TQ.data+( C[1] + (C[4] + C[8]*SQ.data)*SQ.data ) * SQ.data * 1000. + SIGO
    #if is_cube :
    #dens_out.units='kg/m^3'



    #
    #    if to_test eq 1 then begin
    #
    #
    #      dens_test.lbfc=639
    #      dens_test.lbuser(3)=0
    #
    #      help,dens_out,dens_test
    #
    #      !p.multi = [0,2,2]
    #      plot,dens_out,dens_test.data
    #      plot,dens_out - dens_test.data
    #
    #      pm0
    #    endif

    return sigma_dens_out + 1000




def pea_TS(T_in,S_in,gdept,e3t_in,tmask = None,calc_TS_comp = False, zcutoff = 400.):
    #from call_eos import calc_sigma0, calc_sigmai#, calc_albet

    # Create potential energy anomaly.


    nt,nz = T_in.shape[:2]
    # if t and s are not masked arrays turn them into them
    if np.ma.isMA(T_in):
        T = T_in.copy()
        S = S_in.copy()
            
        if tmask is None:
            tmask = T_in.mask==False

    else:

        if tmask is None:
            print('Must pass tmask if T and S are not masked arrays')
            pdb.set_trace()
        else:
            T = np.ma.array(T_in,mask = tmask==False)
            S = np.ma.array(S_in,mask = tmask==False)

    #create masked array of dz mat (e3t)
    e3t = np.ma.array(e3t_in,mask = tmask==False)
    #make index arrays
    ind_0_mat = gdept.data.astype('int').copy() * 0
    ind_1_mat = gdept.data.astype('int').copy() * 0
    ind_2_mat = gdept.data.astype('int').copy() * 0
    ind_3_mat = gdept.data.astype('int').copy() * 0

    for i in range(ind_0_mat.shape[0]):ind_0_mat[i,:,:,:] = i
    for i in range(ind_1_mat.shape[1]):ind_1_mat[:,i,:,:] = i
    for i in range(ind_2_mat.shape[2]):ind_2_mat[:,:,i,:] = i
    for i in range(ind_3_mat.shape[3]):ind_3_mat[:,:,:,i] = i

    #make a weighting array, being the proportion of the grid box begin above a threshold depth (400m)
    ##################################################################################################

    #curr_gdept = gdept.copy()
    curr_gdept = e3t.cumsum(axis = 1)


    # find depths less than the threshold (int index)
    gb_zcut_int = (curr_gdept < zcutoff) + 0             
    # index of box deeper (not shallower!) than threshold depth
    gb_zcut_ind = gb_zcut_int.sum(axis = 1)[0,:,:]  
    # make sure not greater than depth of water
    gb_zcut_ind = np.minimum(gb_zcut_ind,nz-1)        
    # initialise a weighting matrix
    weight_mat = gb_zcut_int.copy() + 0.          

    # find depth above and below threshold
    #   grid boxes shallower than the threshold
    gdept_sh_zcut = curr_gdept[ind_0_mat[:,0,:,:],gb_zcut_ind-1,ind_2_mat[:,0,:,:],ind_3_mat[:,0,:,:]]

    #   grid boxes deeper than the threshold
    gdept_de_zcut = curr_gdept[ind_0_mat[:,0,:,:],np.minimum(gb_zcut_ind,nz)-0,ind_2_mat[:,0,:,:],ind_3_mat[:,0,:,:]]


    # Find the proportion of the grid box that straddles the threhold,that is above the theshold
    prop_ab_zcut = (zcutoff- gdept_sh_zcut) / (gdept_de_zcut - gdept_sh_zcut)
    
    # when the water column is shallowe than the threshold, 
    #   the index of the grid box above and below don't make sense, 
    #   therefore the proportion, and so the weighting >1.
    # Catch this, by setting the weighting of this grid box to zero.
    prop_ab_zcut[gdept_de_zcut<zcutoff] = 1

    # add this proportion to the weighing array
    weight_mat[ind_0_mat[:,0,:,:],gb_zcut_ind,ind_2_mat[:,0,:,:],ind_3_mat[:,0,:,:]] = prop_ab_zcut

    if weight_mat.max() > 1:
        print('')
        print('Error in pea_TS.')
        print('Logic to produce weighting of grid box> depth threshold has failed.')
        print('')
        pdb.set_trace()

    T_mn_zcut = (T*e3t*weight_mat).sum(axis = 1)/(e3t*weight_mat).sum(axis = 1)
    S_mn_zcut = (S*e3t*weight_mat).sum(axis = 1)/(e3t*weight_mat).sum(axis = 1)

    #3d density field
    #rho = calc_sigma0(T,S)
    #from NEMO_nc_slevel_viewer_lib import sw_dens
    rho = sw_dens(T,S) 


    #Density field from depth average T and S
    #rho_mn_lay_zcut = calc_sigma0(T_mn_zcut,S_mn_zcut)
    rho_mn_lay_zcut = sw_dens(T_mn_zcut,S_mn_zcut)
    rho_mn_zcut = rho.copy()
    for zi in range(rho.shape[1]): rho_mn_zcut[:,zi,:,:] = rho_mn_lay_zcut
    drho_zcut = (rho - rho_mn_zcut)



    #3d density field with depth average salinity
    if calc_TS_comp:
        S_mn_zcut_mat = np.tile(S_mn_zcut,(nz,1,1,1)).transpose((1,0,2,3))
        S_mn_zcut_mat.mask = tmask==False
        #rho_mn_zcut_T = calc_sigma0(T,S_mn_zcut_mat)
        rho_mn_zcut_T = sw_dens(T,S_mn_zcut_mat)
        drho_zcut_T = (rho - rho_mn_zcut_T)

    #potential energy anomaly - depth integrated to depth threshold (400m)
    # mistake pea =  9.81*(drho_zcut*gdept*e3t*weight_mat).sum(axis = 1)/(gdept*weight_mat).sum(axis = 1)
    pea =  9.81*(drho_zcut*curr_gdept*e3t*weight_mat).sum(axis = 1)/(e3t*weight_mat).sum(axis = 1)


    if calc_TS_comp:
        pea_T  =         9.81*(drho_zcut_T*curr_gdept*e3t*weight_mat).sum(axis = 1)/(e3t*weight_mat).sum(axis = 1)
        pea_S = pea*(1-pea_T/pea)

        # if PEA == 0, PEA_S becomes masked.
        # catch this, rather than calculate peas from scratch (which is slow)
        pea_S[pea==0] = -pea_T[pea==0]

        # Check that we have not missed any changes in mask size
        if pea.mask.sum()!=pea_S.mask.sum():
            print('peas different size')
            pdb.set_trace()
        
    if calc_TS_comp:
        return pea,pea_T,pea_S
    else:
        return pea




def load_nn_amm15_amm7_wgt(tmpfname_out_amm15_amm7):
    
    amm15_amm7_dict = {}
    rootgrp = Dataset(tmpfname_out_amm15_amm7, 'r')
    amm15_amm7_dict['amm15_amm7_ii'] = rootgrp.variables['amm15_amm7_ii'][:]
    amm15_amm7_dict['amm15_amm7_jj'] = rootgrp.variables['amm15_amm7_jj'][:]
    amm15_amm7_dict['amm15_amm7_dd'] = rootgrp.variables['amm15_amm7_dd'][:]
    amm15_amm7_dict['amm15_amm7_am'] = rootgrp.variables['amm15_amm7_am'][:]
    rootgrp.close()
    
    return amm15_amm7_dict
    
def load_nn_amm7_amm15_wgt(tmpfname_out_amm7_amm15):
    
    amm7_amm15_dict = {}
    rootgrp = Dataset(tmpfname_out_amm7_amm15, 'r')
    amm7_amm15_dict['amm7_amm15_ii'] = rootgrp.variables['amm7_amm15_ii'][:]
    amm7_amm15_dict['amm7_amm15_jj'] = rootgrp.variables['amm7_amm15_jj'][:]
    amm7_amm15_dict['amm7_amm15_dd'] = rootgrp.variables['amm7_amm15_dd'][:]
    amm7_amm15_dict['amm7_amm15_am'] = rootgrp.variables['amm7_amm15_am'][:]
    rootgrp.close()
    
    return amm7_amm15_dict
    

def vector_div(tmpU, tmpV, tmpdx, tmpdy):
    div_out = (np.gradient(tmpU, axis=0)/tmpdx) + (np.gradient(tmpV, axis=1)/tmpdy)

    return div_out



def vector_curl(tmpU, tmpV, tmpdx, tmpdy):
    curl_out = (np.gradient(tmpV, axis=0)/tmpdx) - (np.gradient(tmpU, axis=1)/tmpdy)

    return curl_out








def pycnocline_params(rho_4d,gdept_3d,e3t_3d):




    '''
    N2,Pync_Z,Pync_Th,N2_max = pycnocline_params(data_inst[tmp_datstr][np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])

    should density be in 25kg/m3 or 1025kg/m3... should z be positive, N2 be negative?

    

    gdept_mat_ts = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T
    dz_mat_ts = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T




    '''

    minus_z_3d = gdept_3d
    #pdb.set_trace()
    # vertical density gradient
    drhodz = np.gradient(rho_4d, axis = 1)/(-e3t_3d)
    N2 = drhodz*(-9.81/rho_4d[:,:,:,:])
    '''
    drho =  rho_4d[:,2:,:,:] -  rho_4d[:,:-2,:,:]
    dz = gdept_3d[2:,:,:] - gdept_3d[:-2,:,:]

    drho_dz = drho/dz
    
    # N, Brunt-Vaisala frequency
    # N**2
    N2 = rho_4d.copy()*0*np.ma.masked
    N2[:,1:-1,:,:]  = drho_dz*(-9.81/rho_4d[:,1:-1,:,:])
    N2[:,0,:,:]= N2[:,1,:,:]
    '''

    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018JC014307
    # Equation 14

    
    # Pycnocline Depth:
    #Pync_Z = ((N2*gdept_3d)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1)
    Pync_Z = -((N2*(-minus_z_3d))*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1)
                    
    # Pycnocline thickness:
    #Pync_Th  = np.sqrt(((N2*(gdept_3d-Pync_Z)**2)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1))
    
    
    #Pync_Th  = np.sqrt(((N2*((-minus_z_3d)-Pync_Z)**2)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1))
    Pync_Th  = np.sqrt(((N2*((-minus_z_3d.T)-Pync_Z.T).T**2)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1))
    
    

    # Depth of max Nz
    # find array size
    n_t,n_z, n_j, n_i = rho_4d.shape

    # Make dummy index array
    n_i_mat, n_j_mat = np.meshgrid(range(n_i), range(n_j))


    # find index of maximum N2 depth
    N2_max_arg = N2.argmax(axis = 1)
    N2_max = N2.max(axis = 1)

    # use gdept to calcuate these as a depth
    if n_t <= 1:
        N2_maxZ = gdept_3d[N2_max_arg,np.tile(n_j_mat,(n_t,1,1)),np.tile(n_i_mat,(n_t,1,1))]
    else:
        N2_maxZ = gdept_3d[0,N2_max_arg,np.tile(n_j_mat,(n_t,1,1)),np.tile(n_i_mat,(n_t,1,1))]

    return N2,Pync_Z,Pync_Th,N2_max,N2_maxZ
          
def stream_from_vocetr_eff(v_tr):
    ndim_v_tr = len(v_tr.shape)
    if ndim_v_tr == 2:
        psi = (v_tr[:,::-1].cumsum(axis = 1)[:,::-1])
    elif ndim_v_tr == 3:
        psi = (v_tr[:,:,::-1].cumsum(axis = 2)[:,:,::-1])
    return psi

        
def update_cur_var_grid(var,tmp_datstr,ldi, var_grid, xarr_dict):

    #tmp_cur_var_grid = var_grid(var,tmp_datstr,ldi, var_grid, xarr_dict )
    
    tmp_cur_var_grid_lst = var_grid[var]
    ntmp_cur_var_grid_lst = len(tmp_cur_var_grid_lst)
    
    tmp_cur_var_grid_lst = []

    for tmp_cur_var_grid in var_grid[var]:
        #print(tmp_cur_var_grid)

        # sometime we take T, S and SSH from different files (grid), and compare to a config where they are all in the same file
        # so check where it is?

        check_cur_var_grid = False
        if tmp_cur_var_grid not in xarr_dict[tmp_datstr].keys():
            check_cur_var_grid = True
        else:
            if var not in xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables.keys():
                check_cur_var_grid = True


        if check_cur_var_grid:

            #print('tmp_cur_var_grid',tmp_cur_var_grid)
            #for tmp_var_grid in xarr_dict[tmp_datstr].keys():tmp_var_grid,var,var in xarr_dict[tmp_datstr][tmp_var_grid][ldi].variables.keys(),[ss for ss in xarr_dict[tmp_datstr][tmp_var_grid][ldi].variables.keys()]

            for tmp_var_grid in xarr_dict[tmp_datstr].keys():
                #print('tmp_var_grid',tmp_var_grid)
                if var in xarr_dict[tmp_datstr][tmp_var_grid][ldi].variables.keys():
                    #print('tmp_cur_var_grid before',tmp_cur_var_grid)
                    tmp_cur_var_grid = tmp_var_grid
                    #print('tmp_cur_var_grid after',tmp_cur_var_grid)

        tmp_cur_var_grid_lst.append(tmp_cur_var_grid)
        #print(tmp_cur_var_grid,tmp_cur_var_grid_lst)

    return tmp_cur_var_grid_lst
    



def LBC_iijj_ind(do_LBC_d,LBC_coord_d,var_grid,var, LBC_iijj_threshold = 1):

    tmp_datstr = 'Dataset 1'
    th_d_ind = int(tmp_datstr[8:])
    if do_LBC_d[th_d_ind]:
        
        tmp_LBC_grid = var_grid[tmp_datstr][var][0]
        if tmp_LBC_grid == 'T': tmp_LBC_grid = 'T_1'
        
        LBC_set = int(tmp_LBC_grid[-1])
        LBC_type = tmp_LBC_grid[:-2]

        if LBC_type in ['T','U','V']:
            tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type.lower()]-1
            tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type.lower()]-1
        elif LBC_type in ['T_bt','U_bt','V_bt']:
            tmpLBCnbj =LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
            tmpLBCnbi =LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1

                    
        LBC_dist_mat = np.sqrt((tmpLBCnbj - jj) **2  + (tmpLBCnbi - ii)**2)
        jj = 0
        if LBC_dist_mat.min()<LBC_iijj_threshold:
            ii = LBC_dist_mat.argmin()
        else:
            ii = np.ma.masked

    return ii,jj


def  LBC_regrid_ind_update_one_dataset(do_LBC_d,LBC_coord_d,thd,Dataset_lst,tmp_data_inst,tmp_LBC_data_out,tmp_datstr,tmp_cur_var_grid,grid_dict,var,var_grid):

    th_d_ind = int(tmp_datstr[8:])
    #
    #
    #



    # if the this dataset is LBC
    if do_LBC_d[th_d_ind]:
        try:
            # if the data_inst is not the correct shape, i.e. is likely the LBC rather than the model grid
            if tmp_data_inst.shape[-2:] != grid_dict[tmp_datstr]['gdept'].shape[-2:]:
                
                # LBC_regrid_ind(LBC_coord_d,Dataset_lst,data_inst,grid_dict[tmp_datstr]['gdept'].shape,var_grid)
                # tmp_LBC_data_in = data_inst[tmp_datstr]

                tmp_LBC_grid = tmp_cur_var_grid
                if tmp_LBC_grid == 'T': tmp_LBC_grid = 'T_1'
                
                LBC_set = int(tmp_LBC_grid[-1])
                LBC_type = tmp_LBC_grid[:-2]

            
                tmp_LBC_data_in = tmp_data_inst



                if LBC_type in ['T','U','V']:
                    tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type.lower()] - 1 #[LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]
                    tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type.lower()] - 1#[LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]

                    tmpLBCnbj = tmpLBCnbj//thd[th_d_ind]['dy']
                    tmpLBCnbi = tmpLBCnbi//thd[th_d_ind]['dx']


                    if tmp_LBC_data_out is None:
                        tmp_LBC_data_out = np.ma.zeros(grid_dict[tmp_datstr]['gdept'].shape)*np.ma.masked                                
                    #tmp_LBC_data_out[:,LBC_coord_d[th_d_ind][LBC_set]['nbjt'], LBC_coord_d[th_d_ind][LBC_set]['nbit']] = tmp_LBC_data_in
                    tmp_LBC_data_out[:,tmpLBCnbj,tmpLBCnbi] = tmp_LBC_data_in
                elif LBC_type in ['T_bt','U_bt','V_bt']:
                    #Baltic SSH and U and V have a rim
                    full_rim = (LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()]).size == (tmp_LBC_data_in).size
                    if full_rim:
                        tmpLBCnbj =LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()]-1
                        tmpLBCnbi =LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()]-1
                    else:
                        tmpLBCnbj =LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
                        tmpLBCnbi =LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1


                    tmpLBCnbj = tmpLBCnbj//thd[th_d_ind]['dy']
                    tmpLBCnbi = tmpLBCnbi//thd[th_d_ind]['dx']


                    if tmp_LBC_data_out is None:
                        tmp_LBC_data_out = np.ma.zeros(grid_dict[tmp_datstr]['gdept'].shape[1:])*np.ma.masked  
                    tmp_LBC_data_out[tmpLBCnbj,tmpLBCnbi] = tmp_LBC_data_in
                else:
                    pdb.set_trace()
                tmp_data_inst = tmp_LBC_data_out.copy()
                del(tmp_LBC_data_out)
        except:
            pdb.set_trace()

    return tmp_data_inst


def  LBC_regrid_ind(do_LBC_d,LBC_coord_d,Dataset_lst,data_inst,grid_dict,var,var_grid):
    
    # loop throigh the datasets
    for tmp_datstr in  Dataset_lst:
        th_d_ind = int(tmp_datstr[8:])
        #print('do_LBC_d[th_d_ind]',do_LBC_d[th_d_ind])

        # if the this dataset is LBC
        if do_LBC_d[th_d_ind]:
            try:
                # if the data_inst is not the correct shape, i.e. is likely the LBC rather than the model grid
                if data_inst[tmp_datstr].shape[-2:] != grid_dict[tmp_datstr]['gdept'].shape[-2:]:
                    
                    # LBC_regrid_ind(LBC_coord_d,Dataset_lst,data_inst,grid_dict[tmp_datstr]['gdept'].shape,var_grid)
                    # tmp_LBC_data_in = data_inst[tmp_datstr]

                    tmp_LBC_grid = var_grid[tmp_datstr][var][0]
                    if tmp_LBC_grid == 'T': tmp_LBC_grid = 'T_1'
                    
                    LBC_set = int(tmp_LBC_grid[-1])
                    LBC_type = tmp_LBC_grid[:-2]

                
                    tmp_LBC_data_in = data_inst[tmp_datstr]



                    if LBC_type in ['T','U','V']:
                        tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type.lower()]-1
                        tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type.lower()]-1
                        tmp_LBC_data_out = np.ma.zeros(grid_dict[tmp_datstr]['gdept'].shape)*np.ma.masked                                
                        #tmp_LBC_data_out[:,LBC_coord_d[th_d_ind][LBC_set]['nbjt'], LBC_coord_d[th_d_ind][LBC_set]['nbit']] = tmp_LBC_data_in
                        tmp_LBC_data_out[:,tmpLBCnbj,tmpLBCnbi] = tmp_LBC_data_in
                    elif LBC_type in ['T_bt','U_bt','V_bt']:
                        tmpLBCnbj =LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
                        tmpLBCnbi =LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1

                        tmp_LBC_data_out = np.ma.zeros(grid_dict[tmp_datstr]['gdept'].shape[1:])*np.ma.masked  
                        tmp_LBC_data_out[tmpLBCnbj,tmpLBCnbi] = tmp_LBC_data_in
                    else:
                        pdb.set_trace()
                    data_inst[tmp_datstr] = tmp_LBC_data_out.copy()
                    del(tmp_LBC_data_out)
            except:
                print('Incorrect LBC regridding, possibly using incorrect coord file, or rim width')
                pdb.set_trace()

    return data_inst



def reload_data_instances_time(var,thd,ldi,ti,current_time_datetime_since_1970,time_d,
                               var_d,var_grid,lon_d,lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                               do_LBC = None, do_LBC_d = None,LBC_coord_d = None, EOS_d = None,do_match_time = True):
    #do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d
    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'



    preload_data_ti = ti
    preload_data_var = var
    preload_data_ldi = ldi
    data_inst = {}
    #Dataset_lst = [ss for ss in xarr_dict.keys()]

    if var_grid['Dataset 1'][var][0] == 'WW3':

        if  var in ['wnd_mag']: # ,'barot_div','barot_curl']:
            tmp_var_Ubar = 'uwnd'
            tmp_var_Vbar = 'vwnd'
            #map_dat_2d_U_1 = reload_data_instances('uwnd',thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files)[0]
            #map_dat_2d_V_1 = reload_data_instances('vwnd',thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files)[0]

            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('uwnd',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('vwnd',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            
            #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            if var == 'wnd_mag': 
                for tmp_datstr in Dataset_lst:
                    map_dat_2d_U_1 = data_inst_U[tmp_datstr]
                    map_dat_2d_V_1 = data_inst_V[tmp_datstr]

                    #data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1[tmp_datstr]**2 + map_dat_2d_V_1[tmp_datstr]**2)
                    data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
            #elif  var == 'wnd_div': data_inst[tmp_datstr] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
            #elif var == 'wnd_curl': data_inst[tmp_datstr] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                del(map_dat_2d_U_1)
                del(map_dat_2d_V_1)
            del(data_inst_U)
            del(data_inst_V)
            
        else:
           
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

                tmpdat_inst = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var][0]][ldi].variables[var][ti,:].load()) 

                data_inst[tmp_datstr] = np.ma.array(tmpdat_inst[grid_dict['WW3']['NWS_WW3_nn_ind']],mask = grid_dict['WW3']['AMM15_mask'])[thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
                
                #if you want to use a non masked land value
                #data_inst[tmp_datstr] = data_inst[tmp_datstr].filled(0)
        return data_inst,preload_data_ti,preload_data_var,preload_data_ldi

        #pdb.set_trace()

    start_time_load_inst = datetime.now()

    if var in var_d['d']:
        if var == 'N:P':
            
            #map_dat_N_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['N3n'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #map_dat_P_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['N1p'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #data_inst[tmp_datstr] = map_dat_N_1/map_dat_P_1


            data_inst_N,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('N3n',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_P,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('N1p',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            #tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var][0]][ldi].variables['votemper'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var][0]][ldi].variables['vosaline'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            
            
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

                map_dat_N_1 = data_inst_N[tmp_datstr]
                map_dat_P_1 = data_inst_P[tmp_datstr]
                data_inst[tmp_datstr] = map_dat_N_1/map_dat_P_1

                del(map_dat_N_1)
                del(map_dat_P_1)
            del(data_inst_N)
            del(data_inst_P)

        elif var in ['dUdz']:


            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_U,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                map_dat_3d_U_1 = data_inst_U[tmp_datstr]
                #map_dat_3d_V_1 = data_inst_V[tmp_datstr]

                #map_dat_3d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_U][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #pdb.set_trace()
        
                data_inst[tmp_datstr] = map_dat_3d_U_1
                data_inst[tmp_datstr][0:-1] = map_dat_3d_U_1[0:-1] - map_dat_3d_U_1[1:]

            
                del(map_dat_3d_U_1)
            del(data_inst_U)
            
        elif var in ['dVdz']:
            

            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_V,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                #map_dat_3d_U_1 = data_inst_U[tmp_datstr]
                map_dat_3d_V_1 = data_inst_V[tmp_datstr]
                #map_dat_3d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_V][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                data_inst[tmp_datstr] = map_dat_3d_V_1
                data_inst[tmp_datstr][0:-1] = map_dat_3d_V_1[0:-1] - map_dat_3d_V_1[1:]

                del(map_dat_3d_V_1)
            del(data_inst_V)


        elif var in ['abs_dUdz']:

            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_U,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                map_dat_3d_U_1 = data_inst_U[tmp_datstr]
                #map_dat_3d_V_1 = data_inst_V[tmp_datstr]
                #map_dat_3d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_U][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #pdb.set_trace()
        
                data_inst[tmp_datstr] = map_dat_3d_U_1
                data_inst[tmp_datstr][0:-1] = map_dat_3d_U_1[0:-1] - map_dat_3d_U_1[1:]
                data_inst[tmp_datstr] = np.abs(data_inst[tmp_datstr])
            
                del(map_dat_3d_U_1)
            del(data_inst_U)
            
        elif var in ['abs_dVdz']:
            
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_V,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                #map_dat_3d_U_1 = data_inst_U[tmp_datstr]
                map_dat_3d_V_1 = data_inst_V[tmp_datstr]
                #map_dat_3d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_V][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                data_inst[tmp_datstr] = map_dat_3d_V_1
                data_inst[tmp_datstr][0:-1] = map_dat_3d_V_1[0:-1] - map_dat_3d_V_1[1:]
                data_inst[tmp_datstr] = np.abs(data_inst[tmp_datstr])

                del(map_dat_3d_V_1)
            del(data_inst_V)

        elif var in ['baroc_mag','baroc_div','baroc_curl','baroc_phi']:

            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_U,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_V,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                map_dat_3d_U_1 = data_inst_U[tmp_datstr]
                map_dat_3d_V_1 = data_inst_V[tmp_datstr]

                
                
                #map_dat_3d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_U][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_3d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_V][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                if   var == 'baroc_mag': data_inst[tmp_datstr] = np.sqrt(map_dat_3d_U_1**2 + map_dat_3d_V_1**2)
                elif var == 'baroc_phi': data_inst[tmp_datstr] = 180.*np.arctan2(map_dat_3d_V_1,map_dat_3d_U_1)/np.pi
                elif var == 'baroc_div': data_inst[tmp_datstr] = vector_div(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict[tmp_datstr]['e1t']*thd[th_d_ind]['dx'],grid_dict[tmp_datstr]['e2t']*thd[th_d_ind]['dx'])
                elif var == 'baroc_curl': data_inst[tmp_datstr] = vector_curl(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict[tmp_datstr]['e1t']*thd[th_d_ind]['dx'],grid_dict[tmp_datstr]['e2t']*thd[th_d_ind]['dx'])

                del(map_dat_3d_U_1)
                del(map_dat_3d_V_1)
            del(data_inst_U)
            del(data_inst_V)


        elif var in ['VolTran_e3_mag','VolTran_e3_phi','VolTran_e3_div','VolTran_e3_curl']:
            tmp_var_Ubar = 'uocetr_eff_e3u'
            tmp_var_Vbar = 'vocetr_eff_e3v'
            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Ubar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Vbar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
            
                #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())

                map_dat_2d_U_1 = data_inst_U[tmp_datstr] 
                map_dat_2d_V_1 = data_inst_V[tmp_datstr] 

                
                if   var == 'VolTran_e3_mag': data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
                elif var == 'VolTran_e3_phi': data_inst[tmp_datstr] = 180.*np.arctan2(map_dat_2d_V_1,map_dat_2d_U_1)/np.pi
                elif var == 'VolTran_e3_div': data_inst[tmp_datstr] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                elif var == 'VolTran_e3_curl': data_inst[tmp_datstr] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                del(map_dat_2d_U_1)
                del(map_dat_2d_V_1)
            del(data_inst_U)
            del(data_inst_V)


        elif var in ['VolTran_mag','VolTran_phi','VolTran_div','VolTran_curl']:
            tmp_var_Ubar = 'uocetr_eff'
            tmp_var_Vbar = 'vocetr_eff'
            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Ubar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Vbar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
            
                #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())

                map_dat_2d_U_1 = data_inst_U[tmp_datstr] 
                map_dat_2d_V_1 = data_inst_V[tmp_datstr] 
                
                #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                if   var == 'VolTran_mag': data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
                elif var == 'VolTran_phi': data_inst[tmp_datstr] = 180.*np.arctan2(map_dat_2d_V_1,map_dat_2d_U_1)/np.pi
                elif var == 'VolTran_div': data_inst[tmp_datstr] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                elif var == 'VolTran_curl': data_inst[tmp_datstr] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                del(map_dat_2d_U_1)
                del(map_dat_2d_V_1)
            del(data_inst_U)
            del(data_inst_V)


        elif var in ['barot_mag','barot_phi','barot_div','barot_curl']:

            if 'ubar' in var_d[1]['mat']:
                tmp_var_Ubar = 'ubar'
            elif 'vobtcrtx' in var_d[1]['mat']:
                tmp_var_Ubar = 'vobtcrtx'

            if 'vbar' in var_d[1]['mat']:
                tmp_var_Vbar = 'vbar'
            elif 'vobtcrty' in var_d[1]['mat']:
                tmp_var_Vbar = 'vobtcrty'
            #pdb.set_trace()
            #tmp_var_U_grid = var_grid[tmp_datstr][tmp_var_Ubar]
            #tmp_var_V_grid = var_grid[tmp_datstr][tmp_var_Vbar]



            data_inst_U,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Ubar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Vbar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                map_dat_2d_U_1 = data_inst_U[tmp_datstr]
                map_dat_2d_V_1 = data_inst_V[tmp_datstr]

                
                #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_var_U_grid][ldi].variables[tmp_var_Ubar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_var_V_grid][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                if   var == 'barot_mag':  data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
                elif var == 'barot_phi':  data_inst[tmp_datstr] = 180.*np.arctan2(map_dat_2d_V_1,map_dat_2d_U_1)/np.pi
                elif var == 'barot_div':  data_inst[tmp_datstr] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                elif var == 'barot_curl': data_inst[tmp_datstr] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
                del(map_dat_2d_U_1)
                del(map_dat_2d_V_1)

            del(data_inst_U)
            del(data_inst_V)
    
        elif var in ['StreamFunction']:
            tmp_var_Vbar = 'vocetr_eff'
            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Vbar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                map_dat_2d_V_1 = data_inst_V[tmp_datstr] 
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                
                data_inst[tmp_datstr] = np.ma.array(stream_from_vocetr_eff(map_dat_2d_V_1),mask = map_dat_2d_V_1==0)
                del(map_dat_2d_V_1)
            del(data_inst_V)
    
        elif var in ['StreamFunction_e3']:
            tmp_var_Vbar = 'vocetr_eff_e3v'


            data_inst_V,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time(tmp_var_Vbar,thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            for tmp_datstr in Dataset_lst:
                map_dat_2d_V_1 = data_inst_V[tmp_datstr] 
                #map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            
                data_inst[tmp_datstr] = np.ma.array(stream_from_vocetr_eff(map_dat_2d_V_1),mask = map_dat_2d_V_1==0)
                del(map_dat_2d_V_1)
            del(data_inst_V)
    
        elif var.upper() in ['PEA', 'PEAT','PEAS']:
            data_inst_T,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('votemper',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_S,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('vosaline',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)



            #tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['votemper'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['vosaline'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
                dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

                tmp_T_data_1 = data_inst_T[tmp_datstr]
                tmp_S_data_1 = data_inst_S[tmp_datstr]

                tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(tmp_T_data_1[np.newaxis],tmp_S_data_1[np.newaxis],gdept_mat,dz_mat,calc_TS_comp = True ) 
                if var.upper() == 'PEA':
                    data_inst[tmp_datstr]= tmppea_1[0]
                elif var.upper() == 'PEAT':
                    data_inst[tmp_datstr] = tmppeat_1[0]
                elif var.upper() == 'PEAS':
                    data_inst[tmp_datstr] = tmppeas_1[0]
                del(tmp_T_data_1)
                del(tmp_S_data_1)
            del(data_inst_T)
            del(data_inst_S)


        elif var.upper() in ['RHO','N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
            #tmp_rho = {}
            data_inst_T,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('votemper',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            data_inst_S,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('vosaline',thd,ldi,ti,
                current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)


            #tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['votemper'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            #tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[tmp_datstr][var]][ldi].variables['vosaline'][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            
            
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

                tmp_T_data_1 = data_inst_T[tmp_datstr]
                tmp_S_data_1 = data_inst_S[tmp_datstr]
                
                tmp_rho = sw_dens(tmp_T_data_1,tmp_S_data_1) 
                

                if var.upper() =='RHO'.upper():
                    data_inst[tmp_datstr]=tmp_rho# - 1000.

                elif var.upper() in ['N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
                    
                    N2,Pync_Z,Pync_Th,N2_max,N2_maxz = pycnocline_params(tmp_rho[np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])
                    
                    if var.upper() =='N2'.upper():data_inst[tmp_datstr]=N2[0]
                    elif var.upper() =='Pync_Z'.upper():data_inst[tmp_datstr]=Pync_Z[0]
                    elif var.upper() =='Pync_Th'.upper():data_inst[tmp_datstr]=Pync_Th[0]
                    elif var.upper() =='N2max'.upper():data_inst[tmp_datstr]=N2_max[0]
                del(tmp_T_data_1)
                del(tmp_S_data_1)
                del(tmp_rho)
            del(data_inst_T)
            del(data_inst_S)
        else:
            print("var in var_d['d'], but not encoded")
            pdb.set_trace()




    else:
        #pdb.set_trace()

        #do_match_time = False

        for tmp_datstr in Dataset_lst:
            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

            if tmp_datstr == 'Dataset 1':
                curr_ti = ti
                curr_d_offset = 0
                curr_d_offset_threshold = 1

                curr_load_data = True
            else:
                tmp_grid = var_grid[tmp_datstr][var][0]

                tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
                if len(tmp_cur_var_grid)!=1:
                    pdb.set_trace()
                else:
                    tmp_cur_var_grid = tmp_cur_var_grid[0]

                tmp_datetime_since_1970 = time_d[tmp_datstr][tmp_cur_var_grid]['datetime_since_1970']


                if do_match_time:
                    # Ensure you are loading the same time instances for all data sets, even if they cover different periods
                    # don't load any data for secondary datasets if the times don't match within a tolerance

                    abs_d_offset = np.abs(tmp_datetime_since_1970 - current_time_datetime_since_1970)

                    curr_ti = abs_d_offset.argmin()
                    curr_d_offset = abs_d_offset[curr_ti]

                    curr_d_offset_threshold = np.median(np.diff(tmp_datetime_since_1970))
                    if curr_d_offset_threshold!=curr_d_offset_threshold:
                        curr_d_offset_threshold = 1

                    curr_load_data = curr_d_offset<=curr_d_offset_threshold

                    if curr_load_data == False:
                        print('\nNot Loading data instance for %s as current time is %.2f days from available data, greater than the threshold of %.2f\n\n'%(tmp_datstr,curr_d_offset,curr_d_offset_threshold))

                else:
                    # Allow different times for differnt datasets - helpful for testing.
                    curr_ti = ti
                    curr_d_offset = 0
                    curr_d_offset_threshold = 1
                    if ti<len(tmp_datetime_since_1970):
                        curr_load_data = True
                    else:
                        curr_load_data = False
                    
       
            if curr_load_data:
                #try:


                #pdb.set_trace()

                grid_with_var = var_grid[tmp_datstr][var]
                ngrid_with_var = len(grid_with_var)

                if ngrid_with_var == 1:
                    tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
                    if len(tmp_cur_var_grid)!=1:
                        pdb.set_trace()
                    else:
                        tmp_cur_var_grid = tmp_cur_var_grid[0]
                    if var not in xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables.keys():
                        pdb.set_trace()

                    
                    if var_dim[var] == 3:
                        data_inst[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                        
                    elif var_dim[var] == 4:
                        #pdb.set_trace()
                        data_inst[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                    else:
                        print('var_dim[var] not 3 or 4',var,var_dim[var]  )


                if (ngrid_with_var > 1) & do_LBC_d[th_d_ind]:
                    tmp_data_inst = {}
                    tmp_LBC_data_out = None
                    for tmp_cur_var_grid in grid_with_var:
                    #tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
                    #if var not in xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables.keys():
                    #    pdb.set_trace()

                        if var_dim[var] == 3:
                            tmp_data_inst_gr = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][curr_ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                            
                        elif var_dim[var] == 4:
                            #pdb.set_trace()
                            tmp_data_inst_gr = np.ma.masked_invalid(xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][curr_ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                        else:
                            print('var_dim[var] not 3 or 4',var,var_dim[var]  )

                        
                        #pdb.set_trace()
                        tmp_LBC_data_out = LBC_regrid_ind_update_one_dataset(do_LBC_d,LBC_coord_d,thd,Dataset_lst,tmp_data_inst_gr,tmp_LBC_data_out,tmp_datstr,tmp_cur_var_grid,grid_dict,var,var_grid)
                        del(tmp_data_inst_gr)
                    data_inst[tmp_datstr] = tmp_LBC_data_out
                    #pdb.set_trace()


            else:
                print('reload_data_instants: failed... ti to late?')

                if var_dim[var] == 3:
                    data_inst[tmp_datstr] = np.ma.zeros((xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][0,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].shape))*np.ma.masked
                    
                elif var_dim[var] == 4:
                    data_inst[tmp_datstr] = np.ma.zeros((xarr_dict[tmp_datstr][tmp_cur_var_grid][ldi].variables[var][0,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].shape))*np.ma.masked
                else:
                    print('var_dim[var] not 3 or 4',var,var_dim[var]  )

    print('======================================')
    print('Reloaded data instances for ti = %i, var = %s %s = %s'%(ti,var,datetime.now(),datetime.now() - start_time_load_inst))
    #pdb.set_trace()
    if data_inst['Dataset 1'].mask.all():
        print("\n\n%s: data_inst['Dataset 1'].mask.all()\n\n"%var)
        pdb.set_trace()


    if do_LBC:
        data_inst = LBC_regrid_ind(do_LBC_d,LBC_coord_d,Dataset_lst,data_inst,grid_dict,var,var_grid)

    #pdb.set_trace()
    if EOS_d is None:
        EOS_d = {}
        EOS_d['do_TEOS_EOS_conv'] = False

    if EOS_d['do_TEOS_EOS_conv']:
        if var =='votemper':
            if EOS_d['T']:
                
                data_inst_S,preload_data_ti_T,preload_data_var_T,preload_data_ldi_T= reload_data_instances_time('vosaline',thd,ldi,ti,
                    current_time_datetime_since_1970,time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_2nd_files,
                    do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            
        for tmp_datstr in Dataset_lst:
            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

            dep = grid_dict[tmp_datstr]['gdept']
            lon = lon_d[th_d_ind]
            lat = lat_d[th_d_ind]

            if var =='vosaline':
                if EOS_d['S']:

                    if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                        data_inst[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_S(data_inst[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                        data_inst[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_S(data_inst[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    

            elif var =='votemper':
                if EOS_d['T']:

                    if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                        data_inst[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_T(data_inst[tmp_datstr],data_inst_S[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                        data_inst[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_T(data_inst[tmp_datstr],data_inst_S[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                        


    return data_inst,preload_data_ti,preload_data_var,preload_data_ldi

def EOS_convert_EOS80_2_TEOS10_T(data_in_T, data_in_S, dep, lon, lat, tmp_datstr = None):
    if tmp_datstr is not None: print('Converting EOS80 temperature to TEOS10 for %s'%tmp_datstr,datetime.now())
    import gsw as gsw

    T_conv = gsw.CT_from_pt(data_in_S, data_in_T)
    print('        mean %% diff = %.5f%%'%(100*(T_conv.mean()-data_in_T.mean())/(0.5*T_conv.mean()+data_in_T.mean())))
    return  T_conv

def EOS_convert_TEOS10_2_EOS80_T(data_in_T, data_in_S, dep, lon, lat, tmp_datstr = None):
    if tmp_datstr is not None: print('Converting TEOS10 temperature to EOS80 for %s'%tmp_datstr,datetime.now())

    import gsw as gsw

    T_conv = gsw.pt_from_CT(data_in_S, data_in_T)
    print('        mean %% diff = %.5f%%'%(100*(T_conv.mean()-data_in_T.mean())/(0.5*T_conv.mean()+data_in_T.mean())))

    return  T_conv


def EOS_convert_EOS80_2_TEOS10_S(data_in, dep, lon, lat, tmp_datstr = None):
    if tmp_datstr is not None: print('Converting EOS80 salinity to TEOS10 for %s'%tmp_datstr,datetime.now())
    import gsw as gsw

    pres = gsw.p_from_z(-dep, lat)

    S_conv = gsw.SA_from_SP(data_in.copy(), pres, lon, lat)
    print('        mean %% diff = %.5f%%'%(100*(S_conv.mean()-data_in.mean())/(0.5*S_conv.mean()+data_in.mean())))

    return  S_conv

def EOS_convert_TEOS10_2_EOS80_S(data_in, dep, lon, lat, tmp_datstr = None):
    if tmp_datstr is not None: print('Converting TEOS10 salinity to EOS80 for %s'%tmp_datstr,datetime.now())
    import gsw as gsw

    pres = gsw.p_from_z(-dep, lat)

    S_conv = gsw.SP_from_SA(data_in.copy(), pres, lon, lat)
    print('        mean %% diff = %.5f%%'%(100*(S_conv.mean()-data_in.mean())/(0.5*S_conv.mean()+data_in.mean())))

    return  S_conv

def reload_map_data_comb(var,z_meth,zz,zi, data_inst,var_dim,interp1d_ZwgtT,grid_dict,nav_lon,nav_lat,regrid_params,regrid_meth,thd,configd,Dataset_lst,use_xarray_gdept = False,Sec_regrid = False):

    if var_dim[var] == 3:
        map_dat_dict = reload_map_data_comb_2d(data_inst,regrid_params,regrid_meth ,thd,configd,Dataset_lst,Sec_regrid = Sec_regrid)

    else:
        if z_meth == 'z_slice':
            map_dat_dict = reload_map_data_comb_zmeth_zslice(zz, data_inst,interp1d_ZwgtT,grid_dict,regrid_params,regrid_meth ,thd,configd,Dataset_lst,use_xarray_gdept = use_xarray_gdept,Sec_regrid = Sec_regrid)
        elif z_meth in ['nb','df','zm','zx','zn','zd','zs']:
            map_dat_dict = reload_map_data_comb_zmeth_nb_df_zm_3d(z_meth, data_inst,grid_dict,regrid_params,regrid_meth ,thd,configd,Dataset_lst,Sec_regrid = Sec_regrid)
        elif z_meth in ['ss']:
            map_dat_dict = reload_map_data_comb_zmeth_ss_3d(data_inst,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = Sec_regrid)
        elif z_meth == 'z_index':
            map_dat_dict = reload_map_data_comb_zmeth_zindex(data_inst,zi,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = Sec_regrid)
        else:
            print('z_meth not supported:',z_meth)
            pdb.set_trace()

    map_dat_dict['x'] = nav_lon
    map_dat_dict['y'] = nav_lat

    return map_dat_dict
            


def reload_map_data_comb_2d(data_inst,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = False): # ,
    #Dataset_lst = [ss for ss in data_inst.keys()]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  

    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = data_inst['Dataset 1']
    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        tmp_map_data = data_inst[tmp_datstr]
        map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,tmp_map_data)
        if Sec_regrid: map_dat_dict[tmp_datstr + '_Sec_regrid'] = tmp_map_data


    
    return map_dat_dict



def reload_map_data_comb_zmeth_ss_3d(data_inst,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = False):
    #Dataset_lst = [ss for ss in data_inst.keys()]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  

    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = data_inst['Dataset 1'][0]
    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        tmp_map_data = data_inst[tmp_datstr][0]
        map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,tmp_map_data)
        if Sec_regrid: map_dat_dict[tmp_datstr + '_Sec_regrid'] = tmp_map_data
        #pdb.set_trace()


    return map_dat_dict

def reload_map_data_comb_zmeth_nb_df_zm_3d(z_meth, data_inst,grid_dict,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = False):
    #Dataset_lst = [ss for ss in data_inst.keys()]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    # load files
    map_dat_3d_1 = data_inst['Dataset 1'].copy()
    

    # process onto 2d levels
    if z_meth in ['nb']:
        map_dat_nb_1 = nearbed_int_index_val(map_dat_3d_1)
    elif z_meth == 'df':
        map_dat_ss_1 = map_dat_3d_1[0]
        map_dat_nb_1 = nearbed_int_index_val(map_dat_3d_1)
        map_dat_df_1 = map_dat_ss_1 - map_dat_nb_1
        del(map_dat_ss_1)
        del(map_dat_nb_1)
    elif z_meth == 'zm':
        map_dat_zm_1 = weighted_depth_mean_masked_var(map_dat_3d_1,grid_dict['Dataset 1']['e3t'])
    elif z_meth == 'zx':
        map_dat_zx_1 = map_dat_3d_1.max(axis = 0)
        #map_dat_zx_1 = map_dat_3d_1[:-3].max(axis = 0)
    elif z_meth == 'zn':
        map_dat_zn_1 = map_dat_3d_1.min(axis = 0)
    elif z_meth == 'zd': #  z despike

        #effectively high pass filter the data
        map_dat_3d_1_hpf = map_dat_3d_1[1:-1] - ((map_dat_3d_1[0:-2] + 2*map_dat_3d_1[1:-1] + map_dat_3d_1[2:])/4)
        
        zzzwgt = np.ones((map_dat_3d_1_hpf.shape[0]))
        zzzwgt[1::2] = -1
        map_dat_zd_1 = np.abs((map_dat_3d_1_hpf.T*zzzwgt).T.mean(axis = 0))
        del(map_dat_3d_1_hpf)
    elif z_meth == 'zs':
        map_dat_zs_1 = map_dat_3d_1.std(axis = 0)
    del(map_dat_3d_1)


    if   z_meth == 'nb': map_dat_dict['Dataset 1'] = map_dat_nb_1
    elif z_meth == 'df': map_dat_dict['Dataset 1'] = map_dat_df_1 # map_dat_ss_1 - map_dat_nb_1
    elif z_meth == 'zm': map_dat_dict['Dataset 1'] = map_dat_zm_1
    elif z_meth == 'zx': map_dat_dict['Dataset 1'] = map_dat_zx_1
    elif z_meth == 'zn': map_dat_dict['Dataset 1'] = map_dat_zn_1
    elif z_meth == 'zd': map_dat_dict['Dataset 1'] = map_dat_zd_1
    elif z_meth == 'zs': map_dat_dict['Dataset 1'] = map_dat_zs_1

    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
    
        map_dat_3d_2 = np.ma.masked_invalid(data_inst[tmp_datstr].copy())

        if z_meth == 'nb': 
            map_dat_nb_2 = nearbed_int_index_val(map_dat_3d_2)
        elif z_meth == 'df': 
            map_dat_ss_2 = map_dat_3d_2[0]
            map_dat_nb_2 = nearbed_int_index_val(map_dat_3d_2)
            map_dat_df_2 = map_dat_ss_2 - map_dat_nb_2
        elif z_meth == 'zm': 
            map_dat_zm_2 = weighted_depth_mean_masked_var(map_dat_3d_2,grid_dict[tmp_datstr]['e3t'])
        elif z_meth == 'zx': 
            map_dat_zx_2 = map_dat_3d_2.max(axis = 0)
            #map_dat_zx_2 = map_dat_3d_2[:-3].max(axis = 0)
        elif z_meth == 'zn': 
            map_dat_zn_2 = map_dat_3d_2.min(axis = 0)
        elif z_meth == 'zd': # z despike
            #effectively high pass filter the data
            map_dat_3d_2_hpf = map_dat_3d_2[1:-1] - ((map_dat_3d_2[0:-2] + 2*map_dat_3d_2[1:-1] + map_dat_3d_2[2:])/4)
            
            zzzwgt = np.ones((map_dat_3d_2_hpf.shape[0]))
            zzzwgt[1::2] = -1
            map_dat_zd_2 = np.abs((map_dat_3d_2_hpf.T*zzzwgt).T.mean(axis = 0))
            del(map_dat_3d_2_hpf)
        elif z_meth == 'zs': 
            map_dat_zs_2 = map_dat_3d_2.std(axis = 0)
        del(map_dat_3d_2)
        if   z_meth == 'nb': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_nb_2)
        elif z_meth == 'df': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_df_2)
        elif z_meth == 'zm': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_zm_2)
        elif z_meth == 'zx': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_zx_2)
        elif z_meth == 'zn': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_zn_2)
        elif z_meth == 'zd': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_zd_2)
        elif z_meth == 'zs': map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,map_dat_zs_2)

        if Sec_regrid:
            if   z_meth == 'nb': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_nb_2
            elif z_meth == 'df': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_df_2
            elif z_meth == 'zm': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_zm_2
            elif z_meth == 'zx': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_zx_2
            elif z_meth == 'zn': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_zn_2
            elif z_meth == 'zd': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_zd_2
            elif z_meth == 'zs': map_dat_dict[tmp_datstr + '_Sec_regrid'] = map_dat_zs_2
        
    return map_dat_dict


def reload_map_data_comb_zmeth_zslice(zz, data_inst,interp1d_ZwgtT,grid_dict,regrid_params,regrid_meth,thd,configd,Dataset_lst,use_xarray_gdept = False,Sec_regrid = False):
    #Dataset_lst = [ss for ss in data_inst.keys()]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    if zz not in interp1d_ZwgtT['Dataset 1'].keys():
        interp1d_ZwgtT['Dataset 1'][zz] = interp1dmat_create_weight(grid_dict['Dataset 1']['gdept'],zz,use_xarray_gdept = use_xarray_gdept)


    map_dat_3d_1 = np.ma.masked_invalid(data_inst['Dataset 1'].copy())


    map_dat_dict['Dataset 1'] =  interp1dmat_wgt(np.ma.masked_invalid(map_dat_3d_1),interp1d_ZwgtT['Dataset 1'][zz])

    #if load_2nd_files:
    #    map_dat_3d_2 = np.ma.masked_invalid(data_inst['Dataset 2'])
    #else:
    #    map_dat_3d_2 = map_dat_3d_1


    
    for tmp_datstr in Dataset_lst_secondary:
        map_dat_3d_2 = np.ma.masked_invalid(data_inst[tmp_datstr].copy())
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
    
        if zz not in interp1d_ZwgtT[tmp_datstr].keys(): 
            interp1d_ZwgtT[tmp_datstr][zz] = interp1dmat_create_weight(grid_dict[tmp_datstr]['gdept'],zz,use_xarray_gdept = use_xarray_gdept)
        #pdb.set_trace()
        tmp_map_data = interp1dmat_wgt(np.ma.masked_invalid(map_dat_3d_2),interp1d_ZwgtT[tmp_datstr][zz])
        #pdb.set_trace()
        map_dat_dict[tmp_datstr] = np.ma.masked_invalid(regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,tmp_map_data))
        if Sec_regrid: map_dat_dict[tmp_datstr + '_Sec_regrid'] = tmp_map_data
       

    return map_dat_dict


def reload_map_data_comb_zmeth_zindex(data_inst,zi,regrid_params,regrid_meth,thd,configd,Dataset_lst,Sec_regrid = False):
    #Dataset_lst = [ss for ss in data_inst.keys()]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    map_dat_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][zi])
    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        tmp_map_data = data_inst[tmp_datstr][zi]
        map_dat_dict[tmp_datstr] = np.ma.masked_invalid(regrid_2nd(regrid_params[tmp_datstr],regrid_meth,thd,configd,th_d_ind,tmp_map_data))
        if Sec_regrid: map_dat_dict[tmp_datstr + '_Sec_regrid'] = tmp_map_data


    return map_dat_dict



#def reload_ew_data_comb(ii_in,jj_in, data_inst, nav_lon, nav_lat, grid_dict,n_dim_in,regrid_meth, iijj_ind,Dataset_lst,configd):
def reload_ew_data_comb(ii_in,jj_in, data_inst, lon_d, lat_d, grid_dict,n_dim_in,regrid_meth, iijj_ind,Dataset_lst,configd):
    #Dataset_lst = [ss for ss in data_inst.keys()]
    # nav_lon, nav_lat = lon_d[1], lat_d[1]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  
    '''
    reload the data for the E-W cross-section


    '''
    ii,jj = ii_in,jj_in

    ew_slice_dict = {}



    ew_slice_dict['x'] =  lon_d[1][jj,:].copy()
    ew_slice_dict['y'] =  grid_dict['Dataset 1']['gdept'][:,jj,:].copy()
    ew_slice_dict['Sec Grid'] = {}
    for tmp_datstr in Dataset_lst:
        ew_slice_dict['Sec Grid'][tmp_datstr] = {}
    





    if n_dim_in == 3:   
        ew_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][jj,:]).copy()
    elif n_dim_in == 4:   
        ew_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][:,jj,:]).copy()


    ew_slice_dict['Sec Grid']['Dataset 1']['x'] = ew_slice_dict['x'].copy()
    ew_slice_dict['Sec Grid']['Dataset 1']['y'] = ew_slice_dict['y'].copy()
    ew_slice_dict['Sec Grid']['Dataset 1']['data'] = ew_slice_dict['Dataset 1'].copy()
    ew_slice_dict['Sec Grid']['Dataset 1']['lon'] = lon_d[1][jj,:].copy()
    ew_slice_dict['Sec Grid']['Dataset 1']['lat'] = lat_d[1][jj,:].copy()




    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])



        #if configd[th_d_ind] == configd[1]: #if configd[th_d_ind] is None:

        if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):
            if n_dim_in == 3:   
                ew_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][jj,:]).copy()
            elif n_dim_in == 4:   
                ew_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,jj,:]).copy()
            
            ew_slice_dict['Sec Grid'][tmp_datstr]['x'] = ew_slice_dict['x'].copy()
            ew_slice_dict['Sec Grid'][tmp_datstr]['y'] = ew_slice_dict['y'].copy()
            ew_slice_dict['Sec Grid'][tmp_datstr]['data'] = ew_slice_dict[tmp_datstr].copy()


        else:

            ew_ii_2nd_ind,ew_jj_2nd_ind,ew_bl_jj_ind_final,ew_bl_ii_ind_final,ew_wgt = iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_wgt']

            if regrid_meth == 1:
                if n_dim_in == 3:   
                    tmpdat_ew_slice = np.ma.masked_invalid(data_inst[tmp_datstr][ew_jj_2nd_ind,ew_ii_2nd_ind].T)
                elif n_dim_in == 4: 
                    tmpdat_ew_slice = np.ma.masked_invalid(data_inst[tmp_datstr][:,ew_jj_2nd_ind,ew_ii_2nd_ind].T)  
            elif regrid_meth == 2:  
                if n_dim_in == 3: 
                    tmp_data_inst_2_bl = data_inst[tmp_datstr][ew_bl_jj_ind_final,ew_bl_ii_ind_final]
                    tmp_ew_wgt = ew_wgt.copy()
                    tmp_ew_wgt.mask = tmp_ew_wgt.mask | tmp_data_inst_2_bl.mask
                    tmpdat_ew_slice = ((tmp_data_inst_2_bl* tmp_ew_wgt).sum(axis = 0)/tmp_ew_wgt.sum(axis = 0)).T           
                elif n_dim_in == 4:  
                    tmp_data_inst_2_bl = data_inst[tmp_datstr][:,ew_bl_jj_ind_final,ew_bl_ii_ind_final]
                    tmp_ew_wgt = ew_wgt.copy()
                    tmp_ew_wgt.mask = tmp_ew_wgt.mask | tmp_data_inst_2_bl.mask
                    tmpdat_ew_slice = ((tmp_data_inst_2_bl* tmp_ew_wgt).sum(axis = 1)/tmp_ew_wgt.sum(axis = 0)).T


            if n_dim_in == 3: 
                #tmpdat_ew_gdept=grid_dict[tmp_datstr]['gdept'][:,ew_jj_2nd_ind,ew_ii_2nd_ind].T
                #ew_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0::2])*np.ma.masked
                #for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ew_slice,tmpdat_ew_gdept,ew_slice_dict['y'].T)):ew_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))
                ew_slice_dict[tmp_datstr] = tmpdat_ew_slice
            #pdb.set_trace()
            if n_dim_in == 4: 
                tmpdat_ew_gdept=grid_dict[tmp_datstr]['gdept'][:,ew_jj_2nd_ind,ew_ii_2nd_ind].T
                ew_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0::2])*np.ma.masked
                for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ew_slice,tmpdat_ew_gdept,ew_slice_dict['y'].T)):
                    ew_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))
            if np.ma.is_masked(iijj_ind[tmp_datstr]['jj']):
                ew_slice_dict['Sec Grid'][tmp_datstr]['x'] = ew_slice_dict['x'].copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['y'] = ew_slice_dict['y'].copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['data'] = ew_slice_dict['Dataset 1'].copy()*np.ma.masked
                ew_slice_dict['Sec Grid'][tmp_datstr]['lon'] = ew_slice_dict['x'].copy()*np.ma.masked
                ew_slice_dict['Sec Grid'][tmp_datstr]['lat'] = ew_slice_dict['x'].copy()*np.ma.masked
            else:
                ew_slice_dict['Sec Grid'][tmp_datstr]['x'] = lon_d[th_d_ind][iijj_ind[tmp_datstr]['jj'],:].copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['y'] = grid_dict[tmp_datstr]['gdept'][:,iijj_ind[tmp_datstr]['jj'],:].copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['data'] = np.ma.masked_invalid(data_inst[tmp_datstr].T[:,iijj_ind[tmp_datstr]['jj']].T).copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['lon'] = lon_d[th_d_ind][iijj_ind[tmp_datstr]['jj'],:].copy()
                ew_slice_dict['Sec Grid'][tmp_datstr]['lat'] = lat_d[th_d_ind][iijj_ind[tmp_datstr]['jj'],:].copy()



    #pdb.set_trace()
    return ew_slice_dict

def reload_ns_data_comb(ii_in,jj_in, data_inst, lon_d, lat_d, grid_dict, n_dim_in,regrid_meth,iijj_ind,Dataset_lst,configd):       
#def reload_ns_data_comb(ii_in,jj_in, data_inst, nav_lon, nav_lat, grid_dict, n_dim_in,regrid_meth,iijj_ind,Dataset_lst,configd):        
    #Dataset_lst = [ss for ss in data_inst.keys()]      
    #nav_lon, nav_lat = lon_d[1], lat_d[1]
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')  
    '''
    reload the data for the N-S cross-section

    '''
    ii,jj = ii_in,jj_in



    ns_slice_dict = {}
   
    ns_slice_dict['x'] =  lat_d[1][:,ii].copy()
    ns_slice_dict['y'] =  grid_dict['Dataset 1']['gdept'][:,:,ii].copy()
    ns_slice_dict['Sec Grid'] = {}
    for tmp_datstr in Dataset_lst:
        ns_slice_dict['Sec Grid'][tmp_datstr] = {}
    


    if n_dim_in == 3:   
        ns_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][:,ii]).copy()
    elif n_dim_in == 4:   
        ns_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][:,:,ii]).copy()



    ns_slice_dict['Sec Grid']['Dataset 1']['x'] = ns_slice_dict['x'].copy()
    ns_slice_dict['Sec Grid']['Dataset 1']['y'] = ns_slice_dict['y'].copy()
    ns_slice_dict['Sec Grid']['Dataset 1']['data'] = ns_slice_dict['Dataset 1'].copy()
    ns_slice_dict['Sec Grid']['Dataset 1']['lon'] = lon_d[1][:,ii].copy()
    ns_slice_dict['Sec Grid']['Dataset 1']['lat'] = lat_d[1][:,ii].copy()

    for tmp_datstr in Dataset_lst_secondary:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])




        #if configd[th_d_ind] == configd[1]: #if configd[th_d_ind] is None:
            
        if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):
            #ns_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,:,ii])


            if n_dim_in == 3:   
                ns_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,ii]).copy()
            elif n_dim_in == 4:   
                ns_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,:,ii]).copy()


            ns_slice_dict['Sec Grid'][tmp_datstr]['x'] = ns_slice_dict['x'].copy()
            ns_slice_dict['Sec Grid'][tmp_datstr]['y'] = ns_slice_dict['y'].copy()
            ns_slice_dict['Sec Grid'][tmp_datstr]['data'] = ns_slice_dict[tmp_datstr].copy()




            
        else:
            

            ns_ii_2nd_ind,ns_jj_2nd_ind,ns_bl_jj_ind_final,ns_bl_ii_ind_final,ns_wgt = iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'], iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]   ['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_wgt']
            if regrid_meth == 1:
                if n_dim_in == 3:  
                    tmpdat_ns_slice = np.ma.masked_invalid(data_inst[tmp_datstr][ns_jj_2nd_ind,ns_ii_2nd_ind].T)
                elif n_dim_in == 4:    
                    tmpdat_ns_slice = np.ma.masked_invalid(data_inst[tmp_datstr][:,ns_jj_2nd_ind,ns_ii_2nd_ind].T)
            elif regrid_meth == 2:
                if n_dim_in == 3:  
                    tmp_data_inst_2_bl = data_inst[tmp_datstr][ns_bl_jj_ind_final,ns_bl_ii_ind_final]
                    tmp_ns_wgt = ns_wgt.copy()
                    tmp_ns_wgt.mask = tmp_ns_wgt.mask | tmp_data_inst_2_bl.mask
                    tmpdat_ns_slice = ((tmp_data_inst_2_bl* tmp_ns_wgt).sum(axis = 0)/tmp_ns_wgt.sum(axis = 0)).T
                elif n_dim_in == 4:  
                    tmp_data_inst_2_bl = data_inst[tmp_datstr][:,ns_bl_jj_ind_final,ns_bl_ii_ind_final]
                    tmp_ns_wgt = ns_wgt.copy()
                    tmp_ns_wgt.mask = tmp_ns_wgt.mask | tmp_data_inst_2_bl.mask
                    tmpdat_ns_slice = ((tmp_data_inst_2_bl* tmp_ns_wgt).sum(axis = 1)/tmp_ns_wgt.sum(axis = 0)).T

            if n_dim_in == 3:                  
                #pdb.set_trace()
                #tmpdat_ns_gdept = grid_dict[tmp_datstr]['gdept'][:,ns_jj_2nd_ind,ns_ii_2nd_ind].T
                #ns_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0:2])*np.ma.masked
                #for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ns_slice,tmpdat_ns_gdept,ns_slice_dict['y'].T)):ns_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))
                ns_slice_dict[tmp_datstr] = tmpdat_ns_slice


            elif n_dim_in == 4:  
                tmpdat_ns_gdept = grid_dict[tmp_datstr]['gdept'][:,ns_jj_2nd_ind,ns_ii_2nd_ind].T
                ns_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0:2])*np.ma.masked
                for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ns_slice,tmpdat_ns_gdept,ns_slice_dict['y'].T)):ns_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))
            if np.ma.is_masked(iijj_ind[tmp_datstr]['ii']):
                ns_slice_dict['Sec Grid'][tmp_datstr]['x'] = ns_slice_dict['x'].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['y'] = ns_slice_dict['y'].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['data'] = ns_slice_dict['Dataset 1'].copy()*np.ma.masked
                ns_slice_dict['Sec Grid'][tmp_datstr]['lon'] = ns_slice_dict['x'].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['lat'] = ns_slice_dict['x'].copy()
            else:
                ns_slice_dict['Sec Grid'][tmp_datstr]['x'] = lat_d[th_d_ind][:,iijj_ind[tmp_datstr]['ii']].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['y'] = grid_dict[tmp_datstr]['gdept'][:,:,iijj_ind[tmp_datstr]['ii']].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['data'] = np.ma.masked_invalid(data_inst[tmp_datstr].T[iijj_ind[tmp_datstr]['ii']]).T.copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['lon'] = lon_d[th_d_ind][:,iijj_ind[tmp_datstr]['ii']].copy()
                ns_slice_dict['Sec Grid'][tmp_datstr]['lat'] = lat_d[th_d_ind][:,iijj_ind[tmp_datstr]['ii']].copy()

    return ns_slice_dict



def reload_pf_data_comb(data_inst,var,var_dim,ii,jj,nz,grid_dict,Dataset_lst,configd,iijj_ind):
    
            
    '''
    reload the data for the data_inst
    '''

    pf_dat = {}

    pf_dat['y'] = grid_dict['Dataset 1']['gdept'][:,jj,ii]

    #pdb.set_trace()
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        tmp_jj,tmp_ii = jj,ii
        if configd[th_d_ind] != configd[1]: 
            tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']
                
        if var_dim[var] == 4:
            #pf_dat[tmp_datstr]  = np.ma.masked_invalid(data_inst[tmp_datstr][:,jj,ii])
            #pdb.set_trace()
            if np.ma.is_masked(tmp_jj)|np.ma.is_masked(tmp_ii):
                pf_dat[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
            else:
                tmp_pf = np.ma.masked_invalid(data_inst[tmp_datstr][:,tmp_jj,tmp_ii])
                #pf_dat[tmp_datstr] = np.ma.masked_invalid(np.interp(pf_dat['y'], grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii], tmp_pf))
                pf_dat[tmp_datstr] = np.ma.masked_invalid(np.interp(pf_dat['y'], grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii], tmp_pf.filled(np.nan)))
#            

            #pdb.set_trace()
        else:
            for tmp_datstr in Dataset_lst: pf_dat[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
       
    return pf_dat
    


def reload_time_dist_data_comb_time(var,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,
                              ii_in,jj_in,iijj_ind,nz,ntime,grid_dict,z_meth,zz,zi,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,
                              do_LBC = None, do_LBC_d = None,LBC_coord_d = None, EOS_d = None,do_match_time = True,secdataset_proc = None):       
    #Dataset_lst = [ss for ss in xarr_dict.keys()]   
    # #do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d   
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')    
            
    '''
    reload the data for the timdistmuller plot
    '''
    ii,jj = ii_in,jj_in

    timdist_dat = {}

    for xy in ['x','y']:
        timdist_dat[xy] = {}
        timdist_dat[xy]['t'] = time_datetime.copy()
        if xy == 'x':
            timdist_dat[xy]['z'] = grid_dict['Dataset 1']['gdept'][:,jj_in,:].copy()
            timdist_dat[xy]['x'] =  lon_d[1][jj_in,:].copy()

        elif xy == 'y':
            timdist_dat[xy]['z'] = grid_dict['Dataset 1']['gdept'][:,:,ii_in].copy()
            timdist_dat[xy]['x'] =  lat_d[1][:,ii_in].copy()



        timdist_dat[xy]['Sec Grid'] = {}
        for tmp_datstr in Dataset_lst:
            #if secdataset_proc is not None:
            #    if tmp_datstr == secdataset_proc: continue
            th_d_ind = int(tmp_datstr[8:]) 
            timdist_dat[xy]['Sec Grid'][tmp_datstr] = {}
            tmp_grid = var_grid[tmp_datstr][var][0]
            ############################
            tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
                
            timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'] = time_d[tmp_datstr][tmp_grid]['datetime'].copy()


            #if the same config, extract same point. 
            #if configd[th_d_ind] == configd[1]: 
            if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):            
                ii,jj = ii_in,jj_in    
            #if differnet config
            else:
                # find equivalent iijj coord
                ii,jj = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']


            if xy == 'x':
                timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,jj_in,:].copy()
                timdist_dat[xy]['Sec Grid'][tmp_datstr]['x'] =  lon_d[th_d_ind][jj_in,:].copy()

            elif xy == 'y':
                timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,:,ii_in].copy()
                timdist_dat[xy]['Sec Grid'][tmp_datstr]['x'] =  lat_d[th_d_ind][:,ii_in].copy()

            ############################
            
    for xy in ['x','y']:

        timdist_start = datetime.now()

        """
        if var in deriv_var:
            #pdb.set_trace()
            if var == 'baroc_mag':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr]  = np.sqrt(timdist_dat_U_dict[tmp_datstr]**2 + timdist_dat_V_dict[tmp_datstr]**2)

            elif var == 'baroc_phi':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr]  = 180.*np.arctan2(timdist_dat_V_dict[tmp_datstr],timdist_dat_U_dict[tmp_datstr])/np.pi
        
            elif var == 'dUdz':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                #timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
                #pdb.set_trace()
                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr] = timdist_dat_U_dict[tmp_datstr]
                    timdist_dat[tmp_datstr][0:-1] = timdist_dat[tmp_datstr][0:-1] - timdist_dat[tmp_datstr][1:]

        
            elif var == 'dVdz':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                #timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
                timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr]  = timdist_dat_V_dict[tmp_datstr]
                    timdist_dat[tmp_datstr][0:-1] = timdist_dat[tmp_datstr][0:-1] - timdist_dat[tmp_datstr][1:]

        
            elif var == 'abs_dUdz':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                #timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
                #pdb.set_trace()
                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr] = timdist_dat_U_dict[tmp_datstr]
                    timdist_dat[tmp_datstr][0:-1] = timdist_dat[tmp_datstr][0:-1] - timdist_dat[tmp_datstr][1:]
                    timdist_dat[tmp_datstr] = np.abs(timdist_dat[tmp_datstr])

        
            elif var == 'abs_dVdz':

                tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

                #timdist_dat_U_dict = reload_timdist_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
                timdist_dat_V_dict = reload_timdist_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr]  = timdist_dat_V_dict[tmp_datstr]
                    timdist_dat[tmp_datstr][0:-1] = timdist_dat[tmp_datstr][0:-1] - timdist_dat[tmp_datstr][1:]
                    timdist_dat[tmp_datstr] = np.abs(timdist_dat[tmp_datstr])
        
            elif var == 'rho':
                timdist_dat_T_dict = reload_timdist_data_comb_time('votemper',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                timdist_dat_S_dict = reload_timdist_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                
                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr]  = sw_dens(timdist_dat_T_dict[tmp_datstr].copy(), timdist_dat_S_dict[tmp_datstr].copy())# - 1000
                    timdist_dat['Sec Grid'][tmp_datstr]['data']  = sw_dens(timdist_dat_T_dict['Sec Grid'][tmp_datstr]['data'].copy(), timdist_dat_S_dict['Sec Grid'][tmp_datstr]['data'].copy())# - 1000

            elif var == 'N2':
                try:
                    timdist_dat_T_dict = reload_timdist_data_comb_time('votemper',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                    timdist_dat_S_dict = reload_timdist_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                    for tmp_datstr in Dataset_lst: # _secondary:

                        tmp_jj,tmp_ii = jj,ii

                        if tmp_datstr in Dataset_lst_secondary:
                            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                            if configd[th_d_ind] != configd[1]: 
                                tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']


                        gdept_mat = grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii]
                        dz_mat = grid_dict[tmp_datstr]['e3t'][:,tmp_jj,tmp_ii]
                        nt_ts = timdist_dat_T_dict[tmp_datstr].T.shape[0]

                        gdept_mat_ts = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T
                        dz_mat_ts = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T

                    
                
                        tmp_rho  = sw_dens(timdist_dat_T_dict[tmp_datstr], timdist_dat_S_dict[tmp_datstr])
                        tmp_rho_ts = tmp_rho.T[:,:,np.newaxis,np.newaxis]

                        #pdb.set_trace()
                        tmpN2,tmpPync_Z,tmpPync_Th,tmpN2_max,tmpN2_maxz = pycnocline_params(tmp_rho_ts,gdept_mat_ts,dz_mat_ts )
                        #pdb.set_trace()
                        if var.upper() =='N2'.upper():timdist_dat[tmp_datstr]=tmpN2[:,:,0,0].T

                except:
                    pdb.set_trace()



            else:
                for tmp_datstr in Dataset_lst:
                    timdist_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
        """
        if var in var_mat:

            #tmp_var_grid_ind = 0
            #tmp_cur_var_grid

            for tmp_datstr in Dataset_lst:

                if secdataset_proc is not None:
                    if tmp_datstr != secdataset_proc: continue
                
                th_d_ind = int(tmp_datstr[8:]) 
                #th_d_ind = int(tmp_datstr[8:])

                tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )

                #if tmp_datstr == 'Dataset 1':
                #    timdist_dat_2d = True
                #    if len(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].shape) == 3:
                #        timdist_dat_2d = False


                
                #if the same config, extract same point. 
                #if configd[th_d_ind] == configd[1]: 
                if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):            
                    ii,jj = ii_in,jj_in    
                #if differnet config
                else:
                    # find equivalent iijj coord
                    ii,jj = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']

                #save orig_ii,orig_jj, as gdept not on LBC grid
                orig_ii,orig_jj = ii,jj

                cur_var_grid = None



                if (cur_var_grid is None) | isinstance(cur_var_grid,np.ndarray):
                    cur_var_grid = tmp_cur_var_grid[0]

                if var not in xarr_dict[tmp_datstr][cur_var_grid][ldi].variables.keys():
                    print('reload_timdist_data_comb_time - var no in current grid')
                    pdb.set_trace()

                # Copy to second grid
                timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'] = time_d[tmp_datstr][cur_var_grid]['datetime'].copy()
                #if np.ma.is_masked(ii*jj):
                #    timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size)
                #else:
                #    timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,ii,jj].copy()# [:,thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].copy()
                    
                if tmp_datstr == 'Dataset 1':
                    timdist_dat[xy]['t'] = time_d[tmp_datstr][cur_var_grid]['datetime'].copy()


                tmpnz, tmpnj, tmpni = grid_dict[tmp_datstr]['gdept'][:,:,:].shape

                if np.ma.is_masked(ii*jj):
                    if xy == 'x':
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = np.tile(np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size),(tmpni,1)).T
                    elif xy == 'y':
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = np.tile(np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size),(tmpnj,1)).T
                else:
                    
                    try:
                        #use orig_ii,orig_jj, as gdepth not on LBC grid
                        if xy == 'x':
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,orig_jj,:].copy()
                        elif xy == 'y':
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,:,orig_ii].copy()
                        
                    except:
                        print('get gdepth exception')
                        pdb.set_trace()
                    #timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,jj,ii].load()).T
         







                Sec_Grid_ntime = timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size
                Sec_Grid_nx = timdist_dat[xy]['Sec Grid'][tmp_datstr]['x'].size
                if var_dim[var] == 4:
                    timdist_dat[xy][tmp_datstr] = np.ma.zeros((Sec_Grid_ntime,nz, Sec_Grid_nx))*np.ma.masked
                    timdist_dat[xy]['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((Sec_Grid_ntime,nz, Sec_Grid_nx))*np.ma.masked
                else:
                    timdist_dat[xy][tmp_datstr] = np.ma.zeros((Sec_Grid_ntime, Sec_Grid_nx))*np.ma.masked
                    timdist_dat[xy]['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((Sec_Grid_ntime, Sec_Grid_nx))*np.ma.masked




                #pdb.set_trace()

                if (do_LBC is None)|(do_LBC == False)|(do_LBC & do_LBC_d[th_d_ind] == False):



                    if not np.ma.is_masked(ii*jj): # if it is masked, don't need to do anything as preallocated masked array


                        if xy == 'x':
                            timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][:,jj].load()).T
                            #timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                        elif xy == 'y':
                            timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,:].load()).T
                            #timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                    



                else:

                    if do_LBC:
                        #tmp_datstr = 'Dataset 1'
                        if do_LBC_d[th_d_ind]:
                            cur_var_grid_ii_lst = []
                            for tmp_LBC_grid in tmp_cur_var_grid:
                                if tmp_LBC_grid == 'T': tmp_LBC_grid = 'T_1'

                                tmpii, tmpjj = ii,jj
                                
                                LBC_set = int(tmp_LBC_grid[-1])
                                LBC_type = tmp_LBC_grid[:-2]

                                if LBC_type in ['T','U','V']:
                                    tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type.lower()]-1
                                    tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type.lower()]-1
                                elif LBC_type in ['T_bt','U_bt','V_bt']:
                                    tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
                                    tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1

                                #if var_dim[var] == 3:pdb.set_trace()
                                #pdb.set_trace()
                                if xy == 'x': 
                                    if (orig_jj == tmpLBCnbj).any():
                                        tmpii = 0
                                        #tmpjj = np.where((orig_jj == tmpLBCnbj))[1]
                                        tmpjj = np.where((orig_jj == tmpLBCnbj).T)[0]

                                        if var_dim[var] == 4:
                                            timdist_dat[xy][tmp_datstr][:,:,tmpLBCnbi[orig_jj == tmpLBCnbj]] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,0,tmpjj].load())
                                        elif var_dim[var] == 3:
                                            timdist_dat[xy][tmp_datstr][:,tmpLBCnbj[orig_ii == tmpLBCnbi]] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,0,tmpjj].load())


                                elif xy == 'y':
                                    if (orig_ii == tmpLBCnbi).any():
                                        tmpjj = 0
                                        #tmpii = np.where((orig_ii == tmpLBCnbi))[1]
                                        tmpii = np.where((orig_ii == tmpLBCnbi).T)[0]
                                        
                                        if var_dim[var] == 4:
                                            timdist_dat[xy][tmp_datstr][:,:,tmpLBCnbj[orig_ii == tmpLBCnbi]] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,0,tmpii].load())
                                        elif var_dim[var] == 3:
                                            timdist_dat[xy][tmp_datstr][:,tmpLBCnbj[orig_ii == tmpLBCnbi]] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,0,tmpii].load())


                                timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                    
                


                                #if timdist_dat[xy][tmp_datstr].mask.all():
                                #    print('Time-dist: ',xy,' is all masked')
                                #    pdb.set_trace()
                                    #pdb.set_trace()
                                '''
                                LBC_dist_mat = np.sqrt((tmpLBCnbj - tmpjj) **2  + (tmpLBCnbi - tmpii)**2)
                                if LBC_dist_mat.min()<1:
                                    tmpii = LBC_dist_mat.argmin()
                                else:
                                    tmpii = np.ma.masked
                                '''

                                cur_var_grid_ii_lst.append(tmpii)
                                cur_var_grid_ii_mat = np.ma.array(cur_var_grid_ii_lst)
                            """    
                            jj = 0
                            if cur_var_grid_ii_mat.mask.all():
                                ii = np.ma.masked
                                cur_var_grid = None

                            else:
                                # if point in one grid:
                                if (cur_var_grid_ii_mat.mask == False).sum() == 1:
                                    ii = int(cur_var_grid_ii_mat[~cur_var_grid_ii_mat.mask])
                                    cur_var_grid = np.array(tmp_cur_var_grid)[~cur_var_grid_ii_mat.mask][0]


                                else:
                                    # if point in more than one grid, stop
                                    print(ii,cur_var_grid)
                                    pdb.set_trace()
                            """


                """
                if np.ma.is_masked(ii*jj):
                    timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size)
                    Sec_Grid_ntime = timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size
                    if var_dim[var] == 4:
                        timdist_dat[xy][tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((nz,Sec_Grid_ntime))*np.ma.masked
                    else:
                        timdist_dat[xy][tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((Sec_Grid_ntime))*np.ma.masked
                else:
                    
                    try:
                        #use orig_ii,orig_jj, as gdepth not on LBC grid
                        if xy == 'x':
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,orig_jj,:].copy()
                        elif xy == 'y':
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] = grid_dict[tmp_datstr]['gdept'][:,:,orig_ii].copy()
                        
                    except:
                        print('get gdepth exception')
                        pdb.set_trace()
                    #timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,jj,ii].load()).T
                    
                    if xy == 'x':
                        timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][:,jj].load()).T
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                    elif xy == 'y':
                        timdist_dat[xy][tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,:].load()).T
                        timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                    

                """
                #pdb.set_trace()
                print(tmp_datstr,xy,  timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'].shape, (timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].mask == False).sum())
        
                if do_mask_dict[tmp_datstr]:
                    try:
                    
                        if np.ma.is_masked(ii*jj) == False:
                            
                            if xy == 'x':
                                tmp_mask = grid_dict[tmp_datstr]['tmask'][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,orig_jj,:] == 0
                            elif xy == 'y':
                                tmp_mask = grid_dict[tmp_datstr]['tmask'][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,orig_ii] == 0
                            if var_dim[var] == 3:
                                tmp_mask = tmp_mask[0]
                            timdist_dat[xy][tmp_datstr][:,tmp_mask] = np.ma.masked
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'][:,tmp_mask] = np.ma.masked
                            '''
                            if var_dim[var] == 4:
                                timdist_dat[xy][tmp_datstr][:,tmp_mask] = np.ma.masked
                                timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'][:,tmp_mask] = np.ma.masked
                            else:    
                                timdist_dat[xy][tmp_datstr][:,tmp_mask[0]] = np.ma.masked
                                timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'][:,tmp_mask[0]] = np.ma.masked
                            '''
                            
                    except:
                        print('timdist_time masked exception')
                        pdb.set_trace()


                
                #if the same config, extract same point. 
                #if configd[th_d_ind] == configd[1]: 
                #if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):
                # 
                #    print('')
                #if differnet config
                #else:

                # if the config is different from thefirst one, we need to interpolate the depths (and time)
                if ((configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()))==False:
                    #print('not set for differing configs')
                    #pdb.set_trace()
                    ## find equivalent iijj coord
                    #ii_2nd_ind,jj_2nd_ind = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']

                    

                    # Create a dummy array (effectively copy of Dataset 1)
                    #timdist_dat[xy][tmp_datstr] = np.ma.zeros(xarr_dict['Dataset 1'][var_grid['Dataset 1'][var][0]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].shape[1::-1])*np.ma.masked

                    if not np.ma.is_masked(ii*jj):
                    
                        # extract data for current dataset
                        #tmpdat_timdist = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj,ii].load())
                        #tmpdat_timdist = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,jj,ii].load())
                        #tmpdat_timdist = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load()).T
                    

                        if xy == 'x':
                            tmpdat_timdist = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][:,jj].load()).T
                        elif xy == 'y':
                            tmpdat_timdist = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,:].load()).T
                        
                        timdist_dat[xy][tmp_datstr] =tmpdat_timdist


                    else:
                        print('iijj masked',tmp_datstr)
                        #timdist_dat['Sec Grid'][tmp_datstr]['t'] = timdist_dat['t'].copy()
                        #timdist_dat['Sec Grid'][tmp_datstr]['z']= tmpdat_timdist_gdept.copy()
                        tmpnx = timdist_dat[xy]['Sec Grid'][tmp_datstr]['x'].size


                        if var_dim[var] == 4:#if timdist_dat_2d:
                            timdist_dat[xy][tmp_datstr] = np.ma.zeros((ntime,nz,tmpnx))*np.ma.masked
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = np.ma.zeros((ntime,nz,tmpnx))*np.ma.masked
                        else:
                            timdist_dat[xy][tmp_datstr] = np.ma.zeros((ntime,tmpnx))*np.ma.masked
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = np.ma.zeros((ntime,tmpnx))*np.ma.masked

                    #timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = timdist_dat[xy][tmp_datstr].copy()
                #else:
                #    print('I think we can now natively show timdist_time from other configs')
                #    pdb.set_trace()

                #pdb.set_trace()

                if var_dim[var] == 4:
                    for timdist_ext_ind in [1]: # [0,1]: # original grid and secondary grid

                        if timdist_ext_ind == 0:
                            tmp_timdist_dat = timdist_dat[xy][tmp_datstr].copy()
                            tmp_timdist_z = timdist_dat[xy]['z'].copy()
                            if xy == 'x':
                                #ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,jj_in,:], mask = tmp_timdist_dat[0].mask)
                                ts_e3t_1 = np.ma.array(grid_dict[tmp_datstr]['e3t'][:,jj_in,:], mask = tmp_timdist_dat[0].mask)
                            elif xy == 'y':
                                #ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,:,ii_in], mask = tmp_timdist_dat[0].mask)
                                ts_e3t_1 = np.ma.array(grid_dict[tmp_datstr]['e3t'][:,:,ii_in], mask = tmp_timdist_dat[0].mask)
                        else:
                            tmp_timdist_dat = timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].copy()
                            tmp_timdist_z = timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'].copy()
                            #pdb.set_trace()
                            if xy == 'x':
                                ts_e3t_1 = np.ma.array(grid_dict[tmp_datstr]['e3t'][:,jj_in,:], mask = tmp_timdist_dat[0].mask)
                            elif xy == 'y':
                                ts_e3t_1 = np.ma.array(grid_dict[tmp_datstr]['e3t'][:,:,ii_in], mask = tmp_timdist_dat[0].mask)
                        #pdb.set_trace()
                        if z_meth in ['ss','nb','df','zm','zx','zn','zd','zs']:
                            if z_meth == 'ss':
                                ss_ts_dat_1 = tmp_timdist_dat[:,0,:].copy()
                            elif z_meth == 'nb':
                                timdist_nb_ind_1 = (tmp_timdist_dat[0].mask == False).sum(axis = 0)-1
                                nb_ts_dat_1 = np.ma.array([tmp_timdist_dat[:,nb_i,i_i] for i_i,nb_i in enumerate( timdist_nb_ind_1)]).T
                            elif z_meth == 'df':
                                ss_ts_dat_1 = tmp_timdist_dat[:,0,:].copy()
                                timdist_nb_ind_1 = (tmp_timdist_dat[0].mask == False).sum(axis = 0)-1
                                nb_ts_dat_1 = np.ma.array([tmp_timdist_dat[:,nb_i,i_i] for i_i,nb_i in enumerate( timdist_nb_ind_1)]).T
                                df_ts_dat_1 = ss_ts_dat_1 - nb_ts_dat_1
                            elif z_meth == 'zm':
                                # We are working on the native time grid, but we have interpolated to Dataset 1 depths, so 
                                # we should use e3t from Dataset 1
                                #if xy == 'x':
                                #    ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,jj_in,:], mask = tmp_timdist_dat[0].mask)
                                #elif xy == 'y':
                                #    ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,:,ii_in], mask = tmp_timdist_dat[0].mask)

                                ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum(axis = 0)
                                zm_ts_dat_1 = ((tmp_timdist_dat*ts_dm_wgt_1)).sum(axis =1)
                            elif z_meth == 'zx':
                                zx_ts_dat_1 = tmp_timdist_dat[:].max(axis = 1).copy()
                                #mx_ts_dat_1 = tmp_timdist_dat[:-3].max(axis = 0).copy()
                            elif z_meth == 'zn':
                                zn_ts_dat_1 = tmp_timdist_dat.min(axis = 1).copy()
                            elif z_meth == 'zd': #z depsike
                                #effectively high pass filter the data
                                tmp_timdist_dat_1_hpf = tmp_timdist_dat[:,1:-1] - ((tmp_timdist_dat[:,0:-2] + 2*tmp_timdist_dat[:,1:-1] + tmp_timdist_dat[:,2:])/4)
                                
                                zzzwgt = np.ones((tmp_timdist_dat_1_hpf.shape[1]))
                                zzzwgt[1::2] = -1
                                #zd_ts_dat_1 = np.abs((     tmp_timdist_dat_1_hpf.T*zzzwgt).T.mean(axis = 0))

                                zd_ts_dat_1 = np.abs((tmp_timdist_dat_1_hpf.transpose(0,2,1)*zzzwgt).mean(axis = 2))
                                del(tmp_timdist_dat_1_hpf)
                            elif z_meth == 'zs':
                                zs_ts_dat_1 = tmp_timdist_dat.std(axis = 1).copy()
                            

                            if z_meth == 'ss':
                                tmp_ts_dat_dict_out = ss_ts_dat_1
                            elif z_meth == 'nb':
                                tmp_ts_dat_dict_out = nb_ts_dat_1
                            elif z_meth == 'df':
                                tmp_ts_dat_dict_out = df_ts_dat_1
                            elif z_meth == 'zm':
                                # We are working on the native time grid, but we have interpolated to Dataset 1 depths, so 
                                # we should use e3t from Dataset 1
                                #ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,jj,ii], mask = tmp_timdist_dat[:,0].mask)
                                #ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum()
                                #ts_dat_dict[tmp_datstr] = ((tmp_timdist_dat.T*ts_dm_wgt_1).T).sum(axis = 0)
                                tmp_ts_dat_dict_out = zm_ts_dat_1
                            elif z_meth == 'zx':
                                tmp_ts_dat_dict_out = zx_ts_dat_1
                            elif z_meth == 'zn':
                                tmp_ts_dat_dict_out = zn_ts_dat_1
                            elif z_meth == 'zd':
                                tmp_ts_dat_dict_out = zd_ts_dat_1
                            elif z_meth == 'zs':
                                tmp_ts_dat_dict_out = zs_ts_dat_1

                        elif z_meth == 'z_slice':
                            #for tmp_datstr in Dataset_lst:
                            #tmp_timdist_dat = timdist_dat_dict['Sec Grid'][tmp_datstr]['data']
                            timdist_zi = (np.abs(zz - tmp_timdist_z)).argmin(axis = 0)
                            #ts_dat_dict[tmp_datstr] = tmp_timdist_dat[timdist_zi,:].copy()
                            #tmp_ts_dat_dict_out = tmp_timdist_dat[xy][timdist_zi,:].copy()
                            try:
                                tmp_ts_dat_dict_out = np.ma.array([tmp_timdist_dat[:,zm_i,i_i] for i_i,zm_i in enumerate( timdist_zi)]).T
                            except:
                                pdb.set_trace()

                        elif z_meth == 'z_index':
                            #for tmp_datstr in Dataset_lst:

                            #tmp_timdist_dat = timdist_dat_dict['Sec Grid'][tmp_datstr]['data']

                            #ts_dat_dict[tmp_datstr] = tmp_timdist_dat[zi,:]
                            tmp_ts_dat_dict_out = tmp_timdist_dat[:,zi,:].copy()
                        else:
                            print('reload_ts_data_comb_time z_meth not recognised')
                            pdb.set_trace()

                    
                        if timdist_ext_ind == 0:
                            timdist_dat[xy][tmp_datstr] = tmp_ts_dat_dict_out.copy()
                        else:
                            timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] = tmp_ts_dat_dict_out.copy()
                        del(tmp_ts_dat_dict_out)
                            







        else: # var not in  var_mat or deriv_var


            print('not set for no var')
            pdb.set_trace()
            '''
            for tmp_datstr in Dataset_lst: 
                if var_dim[var] == 4:#if timdist_dat_2d:
                    timdist_dat[xy][tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
                else:
                    timdist_dat[xy][tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
            '''
            
        timdist_stop = datetime.now()
        

        '''
        for tmp_datstr in Dataset_lst:
            if 'data' not in timdist_dat['Sec Grid'][tmp_datstr].keys(): 
                print("Adding timdist_dat['Sec Grid'][tmp_datstr]['data']")
                timdist_dat['Sec Grid'][tmp_datstr]['data'] = timdist_dat[tmp_datstr].copy()
        '''


        #timdist_dat_2d = True
        #if len(timdist_dat['Dataset 1'].shape) == 1:
        #    timdist_dat_2d = False


        """

        ##need to update to check the dimensions are correct

        #print('timdist var_dim[var]',var,var_dim[var])
        # check that the size of dataset1 matchs the time data
        if var_dim[var] == 4: # if timdist_dat is 2d
            if timdist_dat[xy]['Dataset 1'].shape[1] != timdist_dat[xy]['t'].size:
                print("timdist_dat[xy]['Dataset 1'] is 2d, and doesn't match timdist_dat[xy]['t'].size",timdist_dat[xy]['Dataset 1'].shape,timdist_dat[xy]['t'].size )
                pdb.set_trace()
        else: # if timdist_dat[xy] is 1d
            if timdist_dat[xy]['Dataset 1'].size != timdist_dat[xy]['t'].size:
                print("timdist_dat[xy]['Dataset 1'] is 1d, and doesn't match timdist_dat[xy]['t'].size",timdist_dat[xy]['Dataset 1'].shape,timdist_dat[xy]['t'].size )
                pdb.set_trace()




        for tmp_datstr in Dataset_lst:
            print(tmp_datstr, timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'].shape, (timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].mask == False).sum())
                


            if 'data' not in timdist_dat[xy]['Sec Grid'][tmp_datstr].keys():
                pdb.set_trace()

            if var_dim[var] == 4: # if timdist_dat[xy] is 2d
                if timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape[1] != timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size:
                    print("timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] is 2d, and doesn't match timdist_dat[xy]['Sec Grid'][tmp_datstr]['t']",tmp_datstr,timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape,timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size )
                    for tmp_datstr in Dataset_lst:tmp_datstr,timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape
                    pdb.set_trace()
            else: # if timdist_dat[xy] is 1d
                if timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].size != timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size:
                    print("timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'] is 1d, and doesn't match timdist_dat[xy]['Sec Grid'][tmp_datstr]['t']",tmp_datstr,timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape,timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].size )
                    for tmp_datstr in Dataset_lst:tmp_datstr,timdist_dat[xy]['Sec Grid'][tmp_datstr]['t'].shape, timdist_dat[xy]['Sec Grid'][tmp_datstr]['data'].shape
                    pdb.set_trace()
        """
        # temporally regrid the timdist data onto the the Dataset 1
        tmpx_1 = timdist_dat[xy]['t']
        #pdb.set_trace()
        # create a time stamp, time since an origin.
        #if 360 day calendar, timdist_dat[xy]['t'] isn';'t a datetime
        #do_match_time = False

        timdist_date_datetime = True
        if isinstance(timdist_dat[xy]['t'][0],float):
            timdist_date_datetime = False

        # if not 360days, we can use timestamps, otherwise not, 
        if timdist_date_datetime:
            tmp_timestamp_1 = np.array([ss.timestamp() for ss in timdist_dat[xy]['t']])
        else:
            tmp_timestamp_1 = timdist_dat[xy]['t'].copy()

        
        #Estimate a threshold of allowable time differences.     
        curr_d_offset_threshold = np.median(np.diff(tmp_timestamp_1))
        if curr_d_offset_threshold!=curr_d_offset_threshold:
            curr_d_offset_threshold = 86400


        #ntimdistt = tmpx_1# timdist_dat['Dataset 1'].shape[1]
        """
        # we don't need to reinterpolate the depths of other configs or datasets
        #Cyle through the datasets
        for tmp_datstr in Dataset_lst[1:]: 
            #take the time array for that dataset
            tmpx = timdist_dat[xy]['Sec Grid'][tmp_datstr]['t']
            # convert to timestamp, dependng on calendar
            if timdist_date_datetime:
                tmp_timestamp = np.array([ss.timestamp() for ss in tmpx])
            else:
                tmp_timestamp =tmpx.copy()


            # if the 2 time series are the same length, and the same, don't need to regrid. 
            if (timdist_dat[xy]['Sec Grid']['Dataset 1']['t'].size == tmpx.size):
                if (timdist_dat[xy]['Sec Grid']['Dataset 1']['t'] == tmpx).all():
                    continue


            
            tmpdat = timdist_dat[xy]['Sec Grid'][tmp_datstr]['data']
            tmpdat_tint = timdist_dat[xy]['Dataset 1'].copy()*np.ma.masked
            #tmpdat_tint = np.ma.ones(timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'].shape + timdist_dat[xy]['t'].shape)*np.ma.masked


            #pdb.set_trace()
            
            for curr_tind,curr_timestamp in enumerate(tmp_timestamp_1): #curr_tind = 3; curr_timestamp = tmp_timestamp_1[curr_tind]
                #if Dataset 1 timeseries is longer than Dataset 2, 
                #if curr_tind>=tmpdat_tint.shape[1]:
                #    continue


                abs_d_offset = np.abs(tmp_timestamp - curr_timestamp)

                if do_match_time:
                    curr_ti = abs_d_offset.argmin()
                    curr_d_offset = abs_d_offset[curr_ti]

                    curr_load_data = curr_d_offset<curr_d_offset_threshold
                else:
                    curr_ti = curr_tind
                    if curr_ti<len(abs_d_offset):
                        curr_load_data = True
                    else:
                        curr_load_data = False

                try:
                    tmp_vert_interp = True
                    if timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'].size == timdist_dat[xy]['z'].size:
                        if (timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'] == timdist_dat[xy]['z']).all():
                            tmp_vert_interp = False

                    if tmp_vert_interp == False:
                        if curr_load_data:
                            if var_dim[var] == 4: # if timdist_dat[xy] is 2d
                                tmpdat_tint[:,curr_tind] = tmpdat[:,curr_ti]
                            else:
                                tmpdat_tint[curr_tind] = tmpdat[curr_ti]
                    else:
                        if curr_load_data:
                            if var_dim[var] == 4: # if timdist_dat[xy] is 2d

                                #np.ma.masked_invalid(np.interp(timdist_dat[xy]['z'], timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'], tmpdat[:,curr_ti].filled(np.nan)))
                            
                                tmpdat_tint[:,curr_tind] = np.ma.masked_invalid(np.interp(timdist_dat[xy]['z'], timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'], tmpdat[:,curr_ti].filled(np.nan)))
                            else:
                                tmpdat_tint[curr_tind] = np.ma.masked_invalid(np.interp(timdist_dat[xy]['z'], timdist_dat[xy]['Sec Grid'][tmp_datstr]['z'], tmpdat[curr_ti].filled(np.nan)))
                #else:
                #    pdb.set_trace()
                except: 
                    pdb. set_trace()
            timdist_dat[xy][tmp_datstr] = tmpdat_tint

        """
        """
        for tmp_datstr in Dataset_lst[1:]:
            if timdist_dat[xy][tmp_datstr].size != timdist_dat[xy]['Dataset 1'].size:
                print('timdist_dat[xy][' + tmp_datstr +'] size should match timdist_dat[xy][''Dataset 1'']')
                pdb.set_trace()

        """
        """

        #pdb.set_trace()
        if EOS_d is None:
            EOS_d = {}
            EOS_d['do_TEOS_EOS_conv'] = False

        if EOS_d['do_TEOS_EOS_conv']:
            if var =='votemper':
                if EOS_d['T']:
                    
                    timdist_dat_S_dict = reload_timdist_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])




                tmp_jj,tmp_ii = jj,ii

                if tmp_datstr in Dataset_lst_secondary:
                    th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                    if configd[th_d_ind] != configd[1]: 
                        tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']


                #dep = grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii]
                dep = np.tile(grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii],(ntime,1)).T
                lon = lon_d[th_d_ind][tmp_jj,tmp_ii]
                lat = lat_d[th_d_ind][tmp_jj,tmp_ii]

                #pdb.set_trace()

                if var =='vosaline':
                    if EOS_d['S']:

                        if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                            timdist_dat[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_S(timdist_dat[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                        elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                            timdist_dat[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_S(timdist_dat[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                        

                elif var =='votemper':
                    if EOS_d['T']:

                        if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                            timdist_dat[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_T(timdist_dat[tmp_datstr],timdist_dat_S_dict[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                        elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                            timdist_dat[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_T(timdist_dat[tmp_datstr],timdist_dat_S_dict[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                            
        """





    #pdb.set_trace()
    return timdist_dat
          


def reload_hov_data_comb_time(var,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,
                              ii_in,jj_in,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,
                              do_LBC = None, do_LBC_d = None,LBC_coord_d = None, EOS_d = None,do_match_time = True):       
    #Dataset_lst = [ss for ss in xarr_dict.keys()]   
    # #do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d   
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')    
            
    '''
    reload the data for the Hovmuller plot
    '''
    ii,jj = ii_in,jj_in

    hov_dat = {}

    hov_dat['x'] = time_datetime.copy()
    hov_dat['y'] = grid_dict['Dataset 1']['gdept'][:,jj_in,ii_in].copy()
    hov_dat['Sec Grid'] = {}
    for tmp_datstr in Dataset_lst:
        hov_dat['Sec Grid'][tmp_datstr] = {}
        tmp_grid = var_grid[tmp_datstr][var][0]
        
        
        ############################
        tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
            
        hov_dat['Sec Grid'][tmp_datstr]['x'] = time_d[tmp_datstr][tmp_grid]['datetime'].copy()
        ############################

    

    hov_start = datetime.now()


    if var in deriv_var:
        #pdb.set_trace()
        if var == 'baroc_mag':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr]  = np.sqrt(hov_dat_U_dict[tmp_datstr]**2 + hov_dat_V_dict[tmp_datstr]**2)

        elif var == 'baroc_phi':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr]  = 180.*np.arctan2(hov_dat_V_dict[tmp_datstr],hov_dat_U_dict[tmp_datstr])/np.pi
       
        elif var == 'dUdz':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            #hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
            #pdb.set_trace()
            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr] = hov_dat_U_dict[tmp_datstr]
                hov_dat[tmp_datstr][0:-1] = hov_dat[tmp_datstr][0:-1] - hov_dat[tmp_datstr][1:]

       
        elif var == 'dVdz':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            #hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
            hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr]  = hov_dat_V_dict[tmp_datstr]
                hov_dat[tmp_datstr][0:-1] = hov_dat[tmp_datstr][0:-1] - hov_dat[tmp_datstr][1:]

       
        elif var == 'abs_dUdz':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            #hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
            #pdb.set_trace()
            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr] = hov_dat_U_dict[tmp_datstr]
                hov_dat[tmp_datstr][0:-1] = hov_dat[tmp_datstr][0:-1] - hov_dat[tmp_datstr][1:]
                hov_dat[tmp_datstr] = np.abs(hov_dat[tmp_datstr])

       
        elif var == 'abs_dVdz':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            #hov_dat_U_dict = reload_hov_data_comb_time(tmp_var_U,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd)
            hov_dat_V_dict = reload_hov_data_comb_time(tmp_var_V,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr]  = hov_dat_V_dict[tmp_datstr]
                hov_dat[tmp_datstr][0:-1] = hov_dat[tmp_datstr][0:-1] - hov_dat[tmp_datstr][1:]
                hov_dat[tmp_datstr] = np.abs(hov_dat[tmp_datstr])
       
        elif var == 'rho':
            hov_dat_T_dict = reload_hov_data_comb_time('votemper',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            hov_dat_S_dict = reload_hov_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            
            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr]  = sw_dens(hov_dat_T_dict[tmp_datstr].copy(), hov_dat_S_dict[tmp_datstr].copy())# - 1000
                hov_dat['Sec Grid'][tmp_datstr]['data']  = sw_dens(hov_dat_T_dict['Sec Grid'][tmp_datstr]['data'].copy(), hov_dat_S_dict['Sec Grid'][tmp_datstr]['data'].copy())# - 1000

        elif var == 'N2':
            try:
                hov_dat_T_dict = reload_hov_data_comb_time('votemper',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                hov_dat_S_dict = reload_hov_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                for tmp_datstr in Dataset_lst: # _secondary:

                    tmp_jj,tmp_ii = jj,ii

                    if tmp_datstr in Dataset_lst_secondary:
                        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                        if configd[th_d_ind] != configd[1]: 
                            tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']


                    gdept_mat = grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii]
                    dz_mat = grid_dict[tmp_datstr]['e3t'][:,tmp_jj,tmp_ii]
                    nt_ts = hov_dat_T_dict[tmp_datstr].T.shape[0]

                    gdept_mat_ts = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T
                    dz_mat_ts = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T

                
            
                    tmp_rho  = sw_dens(hov_dat_T_dict[tmp_datstr], hov_dat_S_dict[tmp_datstr])
                    tmp_rho_ts = tmp_rho.T[:,:,np.newaxis,np.newaxis]

                    #pdb.set_trace()
                    tmpN2,tmpPync_Z,tmpPync_Th,tmpN2_max,tmpN2_maxz = pycnocline_params(tmp_rho_ts,gdept_mat_ts,dz_mat_ts )
                    #pdb.set_trace()
                    if var.upper() =='N2'.upper():hov_dat[tmp_datstr]=tmpN2[:,:,0,0].T

            except:
                pdb.set_trace()



        else:
            for tmp_datstr in Dataset_lst:
                hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked

    elif var in var_mat:

        #tmp_var_grid_ind = 0
        #tmp_cur_var_grid

        for tmp_datstr in Dataset_lst:
            th_d_ind = int(tmp_datstr[8:]) 

            tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )

            #if tmp_datstr == 'Dataset 1':
            #    hov_dat_2d = True
            #    if len(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].shape) == 3:
            #        hov_dat_2d = False


            
            #if the same config, extract same point. 
            #if configd[th_d_ind] == configd[1]: 
            if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):            
                ii,jj = ii_in,jj_in    
            #if differnet config
            else:
                # find equivalent iijj coord
                ii,jj = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']

            #save orig_ii,orig_jj, as gdept not on LBC grid
            orig_ii,orig_jj = ii,jj

            cur_var_grid = None


            if do_LBC is not None:
                if do_LBC:
                    #tmp_datstr = 'Dataset 1'
                    th_d_ind = int(tmp_datstr[8:])
                    if do_LBC_d[th_d_ind]:
                        cur_var_grid_ii_lst = []
                        for tmp_LBC_grid in tmp_cur_var_grid:
                            if tmp_LBC_grid == 'T': tmp_LBC_grid = 'T_1'

                            tmpii, tmpjj = ii,jj
                            
                            LBC_set = int(tmp_LBC_grid[-1])
                            LBC_type = tmp_LBC_grid[:-2]

                            if LBC_type in ['T','U','V']:
                                tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type.lower()]-1
                                tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type.lower()]-1
                            elif LBC_type in ['T_bt','U_bt','V_bt']:
                                tmpLBCnbj = LBC_coord_d[th_d_ind][LBC_set]['nbj'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
                                tmpLBCnbi = LBC_coord_d[th_d_ind][LBC_set]['nbi'+LBC_type[0].lower()][LBC_coord_d[th_d_ind][LBC_set]['nbr'+LBC_type[0].lower()]==1]-1
                                        
                            LBC_dist_mat = np.sqrt((tmpLBCnbj - tmpjj) **2  + (tmpLBCnbi - tmpii)**2)
                            if LBC_dist_mat.min()<1:
                                tmpii = LBC_dist_mat.argmin()
                            else:
                                tmpii = np.ma.masked

                            cur_var_grid_ii_lst.append(tmpii)
                            cur_var_grid_ii_mat = np.ma.array(cur_var_grid_ii_lst)
                            
                        jj = 0
                        if cur_var_grid_ii_mat.mask.all():
                            ii = np.ma.masked
                            cur_var_grid = None

                        else:
                            # if point in one grid:
                            if (cur_var_grid_ii_mat.mask == False).sum() == 1:
                                ii = int(cur_var_grid_ii_mat[~cur_var_grid_ii_mat.mask])
                                cur_var_grid = np.array(tmp_cur_var_grid)[~cur_var_grid_ii_mat.mask][0]


                            else:
                                # if point in more than one grid, stop
                                print(ii,cur_var_grid)
                                pdb.set_trace()

                            #print(ii,cur_var_grid)


            #tmp_cur_var_grid = tmp_cur_var_grid[0]
            #tmp_cur_var_grid = tmp_cur_var_grid[0]

            if (cur_var_grid is None) | isinstance(cur_var_grid,np.ndarray):
                cur_var_grid = tmp_cur_var_grid[0]

            if var not in xarr_dict[tmp_datstr][cur_var_grid][ldi].variables.keys():
                print('reload_hov_data_comb_time - var no in current grid')
                pdb.set_trace()

            # Copy to second grid
            hov_dat['Sec Grid'][tmp_datstr]['x'] = time_d[tmp_datstr][cur_var_grid]['datetime'].copy()
            #if np.ma.is_masked(ii*jj):
            #    hov_dat['Sec Grid'][tmp_datstr]['y'] = np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size)
            #else:
            #    hov_dat['Sec Grid'][tmp_datstr]['y'] = grid_dict[tmp_datstr]['gdept'][:,ii,jj].copy()# [:,thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].copy()
                
            if tmp_datstr == 'Dataset 1':
                hov_dat['x'] = time_d[tmp_datstr][cur_var_grid]['datetime'].copy()
   
            if np.ma.is_masked(ii*jj):
                hov_dat['Sec Grid'][tmp_datstr]['y'] = np.linspace(0,1,grid_dict[tmp_datstr]['gdept'][:,0,0].size)
                Sec_Grid_ntime = hov_dat['Sec Grid'][tmp_datstr]['x'].size
                if var_dim[var] == 4:
                    hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
                    hov_dat['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((nz,Sec_Grid_ntime))*np.ma.masked
                else:
                    hov_dat[tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
                    hov_dat['Sec Grid'][tmp_datstr]['data']  = np.ma.zeros((Sec_Grid_ntime))*np.ma.masked
            else:
                
                try:
                    #use orig_ii,orig_jj, as gdepth not on LBC grid
                    hov_dat['Sec Grid'][tmp_datstr]['y'] = grid_dict[tmp_datstr]['gdept'][:,orig_jj,orig_ii].copy()# [:,thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].copy()
                except:
                    print('get gdepth exception')
                    pdb.set_trace()
                #hov_dat[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,jj,ii].load()).T
                hov_dat[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
                hov_dat['Sec Grid'][tmp_datstr]['data'] = hov_dat[tmp_datstr].copy() #np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load())
            




            print(tmp_datstr, hov_dat['Sec Grid'][tmp_datstr]['data'].shape, hov_dat['Sec Grid'][tmp_datstr]['x'].shape, hov_dat['Sec Grid'][tmp_datstr]['y'].shape, (hov_dat['Sec Grid'][tmp_datstr]['data'].mask == False).sum())
    
            if do_mask_dict[tmp_datstr]:
                try:
                
                    if np.ma.is_masked(ii*jj) == False:
                        tmp_mask = grid_dict[tmp_datstr]['tmask'][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,orig_jj,orig_ii] == 0
                        if var_dim[var] == 4:
                            hov_dat[tmp_datstr][tmp_mask,:] = np.ma.masked
                            hov_dat['Sec Grid'][tmp_datstr]['data'][tmp_mask,:] = np.ma.masked
                        else:    
                            hov_dat[tmp_datstr][tmp_mask[0]] = np.ma.masked
                            hov_dat['Sec Grid'][tmp_datstr]['data'][tmp_mask[0]] = np.ma.masked
                        
                except:
                    print('hov_time masked exception')
                    pdb.set_trace()


            
            #if the same config, extract same point. 
            #if configd[th_d_ind] == configd[1]: 
            #if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()):
            # 
            #    print('')
            #if differnet config
            #else:

            # if the config is different from thefirst one, we need to interpolate the depths (and time)
            if ((configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper()))==False:
                #pdb.set_trace()
                ## find equivalent iijj coord
                #ii_2nd_ind,jj_2nd_ind = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']

                

                # Create a dummy array (effectively copy of Dataset 1)
                #hov_dat[tmp_datstr] = np.ma.zeros(xarr_dict['Dataset 1'][var_grid['Dataset 1'][var][0]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].shape[1::-1])*np.ma.masked

                if not np.ma.is_masked(ii*jj):
                
                    # extract data for current dataset
                    #tmpdat_hov = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj,ii].load())
                    #tmpdat_hov = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var][:,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,:,jj,ii].load())
                    tmpdat_hov = np.ma.masked_invalid(xarr_dict[tmp_datstr][cur_var_grid][ldi].variables[var].T[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx'],thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']][ii,jj].load()).T
                
                    #hov_dat_2d = True
                    if var_dim[var] == 4:#if hov_dat_2d: #len(tmpdat_hov.shape)==2:
                        # mask if necessary

                        hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked

                        if do_mask_dict[tmp_datstr]:
                            tmp_mask = grid_dict[tmp_datstr]['tmask'][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,orig_jj,orig_ii] == 0
                            tmpdat_hov[tmp_mask,:] = np.ma.masked
                            #pdb.set_trace()

                        #pdb.set_trace()
                        tmpdat_hov_zlev = np.ma.zeros((hov_dat['y'].size,tmpdat_hov.shape[0]))*np.ma.masked


                        # need to regrid vertically to the original grid
                        #   by filling a dummy array, you effectively ensure current dataset is the same size as the new one. 
                        tmpdat_hov_gdept =  grid_dict[tmp_datstr]['gdept'][:,orig_jj,orig_ii]               
                        #for i_i,(tmpdat) in enumerate(tmpdat_hov):hov_dat[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(hov_dat['y'], tmpdat_hov_gdept, tmpdat.filled(np.nan)))
                        for i_i,(tmpdat) in enumerate(tmpdat_hov):tmpdat_hov_zlev[:,i_i] = np.ma.masked_invalid(np.interp(hov_dat['y'], tmpdat_hov_gdept, tmpdat.filled(np.nan)))
                        
                        hov_dat[tmp_datstr]= tmpdat_hov_zlev
                        #hov_dat['Sec Grid'][tmp_datstr]['x']= time_d[tmp_datstr][cur_var_grid]['datetime'].copy()
                        #hov_dat['Sec Grid'][tmp_datstr]['y']= tmpdat_hov_gdept.copy()
                        #hov_dat['Sec Grid'][tmp_datstr]['data']= tmpdat_hov.copy()
                    else: #elif len(tmpdat_hov.shape)==1:

                        if do_mask_dict[tmp_datstr]:
                            tmp_mask = grid_dict[tmp_datstr]['tmask'][0,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,orig_jj,orig_ii] == 0
                            tmpdat_hov[tmp_mask] = np.ma.masked
                        hov_dat[tmp_datstr]= tmpdat_hov
                else:
                    print('iijj masked',tmp_datstr)
                    #hov_dat['Sec Grid'][tmp_datstr]['x'] = hov_dat['x'].copy()
                    #hov_dat['Sec Grid'][tmp_datstr]['y']= tmpdat_hov_gdept.copy()
                    if var_dim[var] == 4:#if hov_dat_2d:
                        hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
                        #hov_dat['Sec Grid'][tmp_datstr]['data'] = np.ma.zeros((nz,ntime))*np.ma.masked
                    else:
                        hov_dat[tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
                        #hov_dat['Sec Grid'][tmp_datstr]['data'] = np.ma.zeros((ntime))*np.ma.masked

                #hov_dat['Sec Grid'][tmp_datstr]['data'] = hov_dat[tmp_datstr].copy()
            #else:
            #    print('I think we can now natively show hov_time from other configs')
            #    pdb.set_trace()
    else: # var not in  var_mat or deriv_var

        for tmp_datstr in Dataset_lst: 
            if var_dim[var] == 4:#if hov_dat_2d:
                hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
            else:
                hov_dat[tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
        
    hov_stop = datetime.now()
    

    '''
    for tmp_datstr in Dataset_lst:
        if 'data' not in hov_dat['Sec Grid'][tmp_datstr].keys(): 
            print("Adding hov_dat['Sec Grid'][tmp_datstr]['data']")
            hov_dat['Sec Grid'][tmp_datstr]['data'] = hov_dat[tmp_datstr].copy()
    '''


    #hov_dat_2d = True
    #if len(hov_dat['Dataset 1'].shape) == 1:
    #    hov_dat_2d = False




    #print('hov var_dim[var]',var,var_dim[var])
    # check that the size of dataset1 matchs the time data
    if var_dim[var] == 4: # if Hov_dat is 2d
        if hov_dat['Dataset 1'].shape[1] != hov_dat['x'].size:
            print("hov_dat['Dataset 1'] is 2d, and doesn't match hov_dat['x'].size",hov_dat['Dataset 1'].shape,hov_dat['x'].size )
            pdb.set_trace()
    else: # if Hov_dat is 1d
        if hov_dat['Dataset 1'].size != hov_dat['x'].size:
            print("hov_dat['Dataset 1'] is 1d, and doesn't match hov_dat['x'].size",hov_dat['Dataset 1'].shape,hov_dat['x'].size )
            pdb.set_trace()




    for tmp_datstr in Dataset_lst:
        print(tmp_datstr, hov_dat['Sec Grid'][tmp_datstr]['data'].shape, hov_dat['Sec Grid'][tmp_datstr]['x'].shape, hov_dat['Sec Grid'][tmp_datstr]['y'].shape, (hov_dat['Sec Grid'][tmp_datstr]['data'].mask == False).sum())
            


        if 'data' not in hov_dat['Sec Grid'][tmp_datstr].keys():
            pdb.set_trace()

        if var_dim[var] == 4: # if Hov_dat is 2d
            if hov_dat['Sec Grid'][tmp_datstr]['data'].shape[1] != hov_dat['Sec Grid'][tmp_datstr]['x'].size:
                print("hov_dat['Sec Grid'][tmp_datstr]['data'] is 2d, and doesn't match hov_dat['Sec Grid'][tmp_datstr]['x']",tmp_datstr,hov_dat['Sec Grid'][tmp_datstr]['data'].shape,hov_dat['Sec Grid'][tmp_datstr]['x'].size )
                for tmp_datstr in Dataset_lst:tmp_datstr,hov_dat['Sec Grid'][tmp_datstr]['x'].shape, hov_dat['Sec Grid'][tmp_datstr]['data'].shape
                pdb.set_trace()
        else: # if Hov_dat is 1d
            if hov_dat['Sec Grid'][tmp_datstr]['data'].size != hov_dat['Sec Grid'][tmp_datstr]['x'].size:
                print("hov_dat['Sec Grid'][tmp_datstr]['data'] is 1d, and doesn't match hov_dat['Sec Grid'][tmp_datstr]['x']",tmp_datstr,hov_dat['Sec Grid'][tmp_datstr]['data'].shape,hov_dat['Sec Grid'][tmp_datstr]['x'].size )
                for tmp_datstr in Dataset_lst:tmp_datstr,hov_dat['Sec Grid'][tmp_datstr]['x'].shape, hov_dat['Sec Grid'][tmp_datstr]['data'].shape
                pdb.set_trace()
    
    # temporally regrid the hov data onto the the Dataset 1
    tmpx_1 = hov_dat['x']
    #pdb.set_trace()
    # create a time stamp, time since an origin.
    #if 360 day calendar, hov_dat['x'] isn';'t a datetime
    #do_match_time = False

    hov_date_datetime = True
    if isinstance(hov_dat['x'][0],float):
        hov_date_datetime = False

    # if not 360days, we can use timestamps, otherwise not, 
    if hov_date_datetime:
        tmp_timestamp_1 = np.array([ss.timestamp() for ss in hov_dat['x']])
    else:
        tmp_timestamp_1 = hov_dat['x'].copy()

    
    #Estimate a threshold of allowable time differences.     
    curr_d_offset_threshold = np.median(np.diff(tmp_timestamp_1))
    if curr_d_offset_threshold!=curr_d_offset_threshold:
        curr_d_offset_threshold = 86400


    #nhovt = tmpx_1# hov_dat['Dataset 1'].shape[1]

    #Cyle through the datasets
    for tmp_datstr in Dataset_lst[1:]: 
        #take the time array for that dataset
        tmpx = hov_dat['Sec Grid'][tmp_datstr]['x']
        # convert to timestamp, dependng on calendar
        if hov_date_datetime:
            tmp_timestamp = np.array([ss.timestamp() for ss in tmpx])
        else:
            tmp_timestamp =tmpx.copy()


        # if the 2 time series are the same length, and the same, don't need to regrid. 
        if (hov_dat['Sec Grid']['Dataset 1']['x'].size == tmpx.size):
            if (hov_dat['Sec Grid']['Dataset 1']['x'] == tmpx).all():
                continue


        
        tmpdat = hov_dat['Sec Grid'][tmp_datstr]['data']
        tmpdat_tint = hov_dat['Dataset 1'].copy()*np.ma.masked
        #tmpdat_tint = np.ma.ones(hov_dat['Sec Grid'][tmp_datstr]['y'].shape + hov_dat['x'].shape)*np.ma.masked


        #pdb.set_trace()
        
        for curr_tind,curr_timestamp in enumerate(tmp_timestamp_1): #curr_tind = 3; curr_timestamp = tmp_timestamp_1[curr_tind]
            #if Dataset 1 timeseries is longer than Dataset 2, 
            #if curr_tind>=tmpdat_tint.shape[1]:
            #    continue


            abs_d_offset = np.abs(tmp_timestamp - curr_timestamp)

            if do_match_time:
                curr_ti = abs_d_offset.argmin()
                curr_d_offset = abs_d_offset[curr_ti]

                curr_load_data = curr_d_offset<curr_d_offset_threshold
            else:
                curr_ti = curr_tind
                if curr_ti<len(abs_d_offset):
                    curr_load_data = True
                else:
                    curr_load_data = False

            try:
                tmp_vert_interp = True
                if hov_dat['Sec Grid'][tmp_datstr]['y'].size == hov_dat['y'].size:
                    if (hov_dat['Sec Grid'][tmp_datstr]['y'] == hov_dat['y']).all():
                        tmp_vert_interp = False

                if tmp_vert_interp == False:
                    if curr_load_data:
                        if var_dim[var] == 4: # if Hov_dat is 2d
                            tmpdat_tint[:,curr_tind] = tmpdat[:,curr_ti]
                        else:
                            tmpdat_tint[curr_tind] = tmpdat[curr_ti]
                else:
                    if curr_load_data:
                        if var_dim[var] == 4: # if Hov_dat is 2d

                            #np.ma.masked_invalid(np.interp(hov_dat['y'], hov_dat['Sec Grid'][tmp_datstr]['y'], tmpdat[:,curr_ti].filled(np.nan)))
                        
                            tmpdat_tint[:,curr_tind] = np.ma.masked_invalid(np.interp(hov_dat['y'], hov_dat['Sec Grid'][tmp_datstr]['y'], tmpdat[:,curr_ti].filled(np.nan)))
                        else:
                            tmpdat_tint[curr_tind] = np.ma.masked_invalid(np.interp(hov_dat['y'], hov_dat['Sec Grid'][tmp_datstr]['y'], tmpdat[curr_ti].filled(np.nan)))
            #else:
            #    pdb.set_trace()
            except: 
                pdb.set_trace()
        hov_dat[tmp_datstr] = tmpdat_tint

    
    for tmp_datstr in Dataset_lst[1:]:
        if hov_dat[tmp_datstr].size != hov_dat['Dataset 1'].size:
            print('hov_dat[' + tmp_datstr +'] size should match hov_dat[''Dataset 1'']')
            pdb.set_trace()





    #pdb.set_trace()
    if EOS_d is None:
        EOS_d = {}
        EOS_d['do_TEOS_EOS_conv'] = False

    if EOS_d['do_TEOS_EOS_conv']:
        if var =='votemper':
            if EOS_d['T']:
                
                hov_dat_S_dict = reload_hov_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

            
        for tmp_datstr in Dataset_lst:
            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])




            tmp_jj,tmp_ii = jj,ii

            if tmp_datstr in Dataset_lst_secondary:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                if configd[th_d_ind] != configd[1]: 
                    tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']


            #dep = grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii]
            dep = np.tile(grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii],(ntime,1)).T
            lon = lon_d[th_d_ind][tmp_jj,tmp_ii]
            lat = lat_d[th_d_ind][tmp_jj,tmp_ii]

            #pdb.set_trace()

            if var =='vosaline':
                if EOS_d['S']:

                    if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                        hov_dat[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_S(hov_dat[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                        hov_dat[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_S(hov_dat[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    

            elif var =='votemper':
                if EOS_d['T']:

                    if EOS_d[th_d_ind] == 'TEOS10_2_EOS80':
                        hov_dat[tmp_datstr] = EOS_convert_TEOS10_2_EOS80_T(hov_dat[tmp_datstr],hov_dat_S_dict[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                    elif EOS_d[th_d_ind] == 'EOS80_2_TEOS10':
                        hov_dat[tmp_datstr] = EOS_convert_EOS80_2_TEOS10_T(hov_dat[tmp_datstr],hov_dat_S_dict[tmp_datstr], dep, lon, lat,tmp_datstr = tmp_datstr)
                        

    return hov_dat
          


def reload_ts_data_comb_time(var,var_dim,var_grid,ii_in,jj_in,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,lon_d,lat_d,
                             xarr_dict,do_mask_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,Dataset_lst,load_2nd_files,
                             do_LBC = None, do_LBC_d = None,LBC_coord_d = None, EOS_d = None,do_match_time = True):
    #do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d
    #Dataset_lst = [ss for ss in hov_dat_dict.keys()]      
    #Dataset_lst.remove('x')       
    #Dataset_lst.remove('y')    


    ii,jj = ii_in,jj_in
    Dataset_lst_secondary = Dataset_lst.copy()
    if 'Dataset 1' in Dataset_lst_secondary: Dataset_lst_secondary.remove('Dataset 1')    
    #        
    ts_dat_dict = {}
    ts_dat_dict['x'] = time_datetime
            
    ts_dat_dict['Sec Grid'] = {}
    for tmp_datstr in Dataset_lst:
        ts_dat_dict['Sec Grid'][tmp_datstr] = {}
        tmp_grid = var_grid[tmp_datstr][var][0]
        tmp_cur_var_grid = update_cur_var_grid(var,tmp_datstr,ldi, var_grid[tmp_datstr], xarr_dict )
        #if len(tmp_cur_var_grid)!=1:
        #    #pdb.set_trace()
        #    
        #    tmp_cur_var_grid = tmp_cur_var_grid[0]
        #else:
        #    tmp_cur_var_grid = tmp_cur_var_grid[0]
            
        ts_dat_dict['Sec Grid'][tmp_datstr]['x'] = time_d[tmp_datstr][tmp_grid]['datetime']




    if var_grid['Dataset 1'][var][0] == 'WW3':
        if var in ['wnd_mag']:
            #pdb.set_trace()

            #ts_dat_U_1 = reload_ts_data_comb_time('uwnd',var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,xarr_dict,do_mask_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,Dataset_lst,load_2nd_files)
            #ts_dat_V_1 = reload_ts_data_comb_time('vwnd',var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,xarr_dict,do_mask_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,Dataset_lst,load_2nd_files)


            ts_dat_U_1 = reload_ts_data_comb_time('uwnd',var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,lon_d,lat_d,xarr_dict,do_mask_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,Dataset_lst,load_2nd_files,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            ts_dat_V_1 = reload_ts_data_comb_time('vwnd',var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,lon_d,lat_d,xarr_dict,do_mask_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,Dataset_lst,load_2nd_files,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            
            
            if var == 'wnd_mag': 
                for tmp_datstr in Dataset_lst: ts_dat_dict[tmp_datstr] = np.sqrt(ts_dat_U_1[tmp_datstr]**2 + ts_dat_V_1[tmp_datstr]**2)
            del(ts_dat_U_1)
            del(ts_dat_V_1)

        else:
            for tmp_datstr in Dataset_lst: # _secondary:
                tmp_jj,tmp_ii = jj,ii
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                if tmp_datstr in Dataset_lst_secondary:
                    #th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                    #pdb.set_trace()
                    if configd[th_d_ind] != configd[1]: #if configd[th_d_ind] is not None:
                        tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']
                
                tmpind = grid_dict['WW3']['NWS_WW3_nn_ind'][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][tmp_jj,tmp_ii]

                #print('WW3 ind:',tmp_jj,tmp_ii,tmpind)
                if grid_dict['WW3']['AMM15_mask'][tmp_jj,tmp_ii]:
                    ts_dat_dict[tmp_datstr] = np.ma.zeros((xarr_dict[tmp_datstr]['WW3'][ldi].variables[var].shape[0]))*np.ma.masked
                else:
                    ts_dat_dict[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['WW3'][ldi].variables[var][:,tmpind].load())

    
        for tmp_datstr in Dataset_lst:
            ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = ts_dat_dict[tmp_datstr].copy()
        #pdb.set_trace()

        return ts_dat_dict 

    if var_dim[var] == 3:
        if var in deriv_var:
            if var.upper() in ['PEA', 'PEAT','PEAS','Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
                try:

                    hov_dat_T_dict = reload_hov_data_comb_time('votemper',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                    hov_dat_S_dict = reload_hov_data_comb_time('vosaline',var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                    for tmp_datstr in Dataset_lst: # _secondary:

                        tmp_jj,tmp_ii = jj,ii

                        if tmp_datstr in Dataset_lst_secondary:
                            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                            if configd[th_d_ind] != configd[1]: 
                                tmp_jj,tmp_ii = iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']


                        #reload_hov interpolates it onto the depths of the dataset 1 profile, so 
                        #   need to use Dataset 1 for gdept and dz_mat
                        #   and to use there original iijj

                        #gdept_mat = grid_dict[tmp_datstr]['gdept'][:,tmp_jj,tmp_ii]
                        #dz_mat = grid_dict[tmp_datstr]['e3t'][:,tmp_jj,tmp_ii]

                        gdept_mat = grid_dict['Dataset 1']['gdept'][:,jj,ii]
                        dz_mat = grid_dict['Dataset 1']['e3t'][:,jj,ii]
                        
                        nt_ts = hov_dat_T_dict[tmp_datstr].T.shape[0]

                        gdept_mat_ts = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T
                        dz_mat_ts = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_ts)).T

                        if var.upper() in ['PEA', 'PEAT','PEAS']:
                            tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(hov_dat_T_dict[tmp_datstr].T[:,:,np.newaxis,np.newaxis],
                                                                    hov_dat_S_dict[tmp_datstr].T[:,:,np.newaxis,np.newaxis],
                                                                    gdept_mat_ts,dz_mat_ts,calc_TS_comp = True )
                            if var.upper() == 'PEA':
                                ts_dat_dict[tmp_datstr] = tmppea_1[:,0,0] 
                            elif var.upper() == 'PEAT':
                                ts_dat_dict[tmp_datstr] = tmppeat_1[:,0,0] 
                            elif var.upper() == 'PEAS':
                                ts_dat_dict[tmp_datstr] = tmppeas_1[:,0,0] 

                        elif var.upper() in ['Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
                    
                            tmp_rho  = sw_dens(hov_dat_T_dict[tmp_datstr], hov_dat_S_dict[tmp_datstr])
                            tmp_rho_ts = tmp_rho.T[:,:,np.newaxis,np.newaxis]

                            #pdb.set_trace()
                            #tmpN2,tmpPync_Z,tmpPync_Th,tmpN2_max,tmpN2_maxz = pycnocline_params_time(tmp_rho_ts,gdept_mat_ts,dz_mat_ts )
                            tmpN2,tmpPync_Z,tmpPync_Th,tmpN2_max,tmpN2_maxz = pycnocline_params(tmp_rho_ts,gdept_mat_ts,dz_mat_ts )
                        
                            if var.upper() =='N2'.upper():ts_dat_dict[tmp_datstr]=tmpN2[:,0,0] 
                            elif var.upper() =='Pync_Z'.upper():ts_dat_dict[tmp_datstr]=tmpPync_Z[:,0,0] 
                            elif var.upper() =='Pync_Th'.upper():ts_dat_dict[tmp_datstr]=tmpPync_Th[:,0,0] 
                            elif var.upper() =='N2max'.upper():ts_dat_dict[tmp_datstr]=tmpN2_max[:,0,0] 

               
                except:
                    print('Problem with reload_ts_data_comb derived 2d vars')
                    pdb.set_trace()

            else:

                for tmp_datstr in Dataset_lst: ts_dat_dict[tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
                

        else:
            
            hov_dat_dict = reload_hov_data_comb_time(var,var_mat,var_grid,var_dim,deriv_var,ldi,thd,time_datetime,time_d,ii,jj,iijj_ind,nz,ntime,grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_2nd_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
            #pdb.set_trace()

    
            for tmp_datstr in Dataset_lst[1:]:
                if hov_dat_dict[tmp_datstr].size != hov_dat_dict['Dataset 1'].size:
                    print('hov_dat[' + tmp_datstr +'] size should match hov_dat[''Dataset 1'']')
                    pdb.set_trace()



            for tmp_datstr in Dataset_lst:  ts_dat_dict[tmp_datstr] = hov_dat_dict[tmp_datstr].copy()
            for tmp_datstr in Dataset_lst:  ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = hov_dat_dict['Sec Grid'][tmp_datstr]['data'].copy()
            #tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']


    elif var_dim[var] == 4:



        #tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']

        for hov_ext_ind in range(2):

            for tmp_datstr in Dataset_lst:
                if hov_ext_ind == 0:
                    tmp_hov_dat = hov_dat_dict[tmp_datstr]
                else:
                    tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']

                #print('reload ts from hov 3d',hov_ext_ind,tmp_datstr)
                #tmp_hov_dat = hov_dat_dict[tmp_datstr]
                #this should be done on the native time grid
                #tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']


                if z_meth in ['ss','nb','df','zm','zx','zn','zd','zs']:
                    if z_meth == 'ss':
                        ss_ts_dat_1 = tmp_hov_dat[0,:].ravel()
                    elif z_meth == 'nb':
                        hov_nb_ind_1 = (tmp_hov_dat[:,0].mask == False).sum()-1
                        nb_ts_dat_1 = tmp_hov_dat[hov_nb_ind_1,:].ravel()
                    elif z_meth == 'df':
                        ss_ts_dat_1 = tmp_hov_dat[0,:].ravel()
                        hov_nb_ind_1 = (tmp_hov_dat[:,0].mask == False).sum()-1
                        nb_ts_dat_1 = tmp_hov_dat[hov_nb_ind_1,:].ravel()
                        df_ts_dat_1 = ss_ts_dat_1 - nb_ts_dat_1
                    elif z_meth == 'zm':
                        # We are working on the native time grid, but we have interpolated to Dataset 1 depths, so 
                        # we should use e3t from Dataset 1
                        ts_e3t_1 = np.ma.array(grid_dict[ 'Dataset 1' ]['e3t'][:,jj,ii], mask = tmp_hov_dat[:,0].mask)
                        ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum()
                        zm_ts_dat_1 = ((tmp_hov_dat.T*ts_dm_wgt_1).T).sum(axis = 0)
                    elif z_meth == 'zx':
                        zx_ts_dat_1 = tmp_hov_dat[:].max(axis = 0).ravel()
                        #mx_ts_dat_1 = tmp_hov_dat[:-3].max(axis = 0).ravel()
                    elif z_meth == 'zn':
                        zn_ts_dat_1 = tmp_hov_dat.min(axis = 0).ravel()
                    elif z_meth == 'zd': #z depsike
                        #effectively high pass filter the data
                        tmp_hov_dat_1_hpf = tmp_hov_dat[1:-1] - ((tmp_hov_dat[0:-2] + 2*tmp_hov_dat[1:-1] + tmp_hov_dat[2:])/4)
                        
                        zzzwgt = np.ones((tmp_hov_dat_1_hpf.shape[0]))
                        zzzwgt[1::2] = -1
                        zd_ts_dat_1 = np.abs((tmp_hov_dat_1_hpf.T*zzzwgt).T.mean(axis = 0))
                        del(tmp_hov_dat_1_hpf)
                    elif z_meth == 'zs':
                        zs_ts_dat_1 = tmp_hov_dat.std(axis = 0).ravel()
                    

                    if z_meth == 'ss':
                        tmp_ts_dat_dict_out = ss_ts_dat_1
                    elif z_meth == 'nb':
                        tmp_ts_dat_dict_out = nb_ts_dat_1
                    elif z_meth == 'df':
                        tmp_ts_dat_dict_out = df_ts_dat_1
                    elif z_meth == 'zm':
                        # We are working on the native time grid, but we have interpolated to Dataset 1 depths, so 
                        # we should use e3t from Dataset 1
                        #ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,jj,ii], mask = tmp_hov_dat[:,0].mask)
                        #ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum()
                        #ts_dat_dict[tmp_datstr] = ((tmp_hov_dat.T*ts_dm_wgt_1).T).sum(axis = 0)
                        tmp_ts_dat_dict_out = zm_ts_dat_1
                    elif z_meth == 'zx':
                        tmp_ts_dat_dict_out = zx_ts_dat_1
                    elif z_meth == 'zn':
                        tmp_ts_dat_dict_out = zn_ts_dat_1
                    elif z_meth == 'zd':
                        tmp_ts_dat_dict_out = zd_ts_dat_1
                    elif z_meth == 'zs':
                        tmp_ts_dat_dict_out = zs_ts_dat_1

                elif z_meth == 'z_slice':
                    #for tmp_datstr in Dataset_lst:
                    #tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']
                    hov_zi = (np.abs(zz - hov_dat_dict['y'])).argmin()
                    #ts_dat_dict[tmp_datstr] = tmp_hov_dat[hov_zi,:].ravel()
                    tmp_ts_dat_dict_out = tmp_hov_dat[hov_zi,:].ravel()


                elif z_meth == 'z_index':
                    #for tmp_datstr in Dataset_lst:

                    #tmp_hov_dat = hov_dat_dict['Sec Grid'][tmp_datstr]['data']

                    #ts_dat_dict[tmp_datstr] = tmp_hov_dat[zi,:]
                    tmp_ts_dat_dict_out = tmp_hov_dat[zi,:]
                else:
                    print('reload_ts_data_comb_time z_meth not recognised')
                    pdb.set_trace()

            
                if hov_ext_ind == 0:
                    ts_dat_dict[tmp_datstr] = tmp_ts_dat_dict_out.copy()
                else:
                    ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = tmp_ts_dat_dict_out.copy()
                del(tmp_ts_dat_dict_out)
                    

    #`pdb.set_trace()
    

    #ts_dat_dict['Sec Grid'][tmp_datstr]['x'] = time_d[tmp_datstr][tmp_grid]['datetime']
    for tmp_datstr in Dataset_lst:
        ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = ts_dat_dict[tmp_datstr].copy()



    #pdb.set_trace()
    for tmp_datstr in Dataset_lst[:]: 
        if ts_dat_dict['Sec Grid'][tmp_datstr]['x'].size !=  ts_dat_dict['Sec Grid'][tmp_datstr]['data'].size: 


            #pdb.set_trace()

            ts_dat_dict['Sec Grid'][tmp_datstr]['x'] = hov_dat_dict['x']
            ts_dat_dict['x'] = hov_dat_dict['x']

            #pdb.set_trace()







    
    ts_dat_2d = True
    if len(ts_dat_dict['Dataset 1'].shape) == 1:
        ts_dat_2d = False




    #print('ts_dat_2d',var,ts_dat_2d)

    if ts_dat_2d:
        pdb.set_trace()
    # check that the size of dataset1 matchs the time data
    if ts_dat_2d: # if ts_dat_dict is 2d
        if ts_dat_dict['Dataset 1'].shape[1] != ts_dat_dict['x'].size:
            print("ts_dat_dict['Dataset 1'] is 2d, and doesn't match ts_dat_dict['x'].size",ts_dat_dict['Dataset 1'].shape,ts_dat_dict['x'].size )
            pdb.set_trace()
    else: # if ts_dat_dict is 1d
        if ts_dat_dict['Dataset 1'].size != ts_dat_dict['x'].size:
            print("ts_dat_dict['Dataset 1'] is 1d, and doesn't match ts_dat_dict['x'].size",ts_dat_dict['Dataset 1'].shape,ts_dat_dict['x'].size )
            pdb.set_trace()




    for tmp_datstr in Dataset_lst:
        if ts_dat_2d: # if ts_dat_dict is 2d
            if ts_dat_dict['Sec Grid'][tmp_datstr]['data'].shape[1] != ts_dat_dict['Sec Grid'][tmp_datstr]['x'].size:
                print("ts_dat_dict['Sec Grid'][tmp_datstr]['data'] is 2d, and doesn't match ts_dat_dict['Sec Grid'][tmp_datstr]['x']",tmp_datstr,ts_dat_dict['Sec Grid'][tmp_datstr]['data'].shape,ts_dat_dict['Sec Grid'][tmp_datstr]['x'].size )
                pdb.set_trace()
        else: # if ts_dat_dict is 1d
            if ts_dat_dict['Sec Grid'][tmp_datstr]['data'].size != ts_dat_dict['Sec Grid'][tmp_datstr]['x'].size:
                print("ts_dat_dict['Sec Grid'][tmp_datstr]['data'] is 2d, and doesn't match ts_dat_dict['Sec Grid'][tmp_datstr]['x']",tmp_datstr,ts_dat_dict['Sec Grid'][tmp_datstr]['data'].shape,ts_dat_dict['Sec Grid'][tmp_datstr]['x'].size )
                pdb.set_trace()
    



    return ts_dat_dict 




def add_derived_vars(var_d,var_dim, var_grid,Dataset_lst):

    print(Dataset_lst)
    for tmp_datstr in Dataset_lst: # xarr_dict.keys():
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

        if ('votemper' in var_d[th_d_ind]['mat']) & ('vosaline' in var_d[th_d_ind]['mat']):
            for ss in ['pea','peat','peas']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    #pdb.set_trace()
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)



            for ss in ['rho']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 4
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)


            for ss in ['N2']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 4
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)


            for ss in ['N2max']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)


            for ss in ['Pync_Z']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)


            for ss in ['Pync_Th']:
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['votemper'] # 'T'
                    var_d['d'].append(ss)


        if ('vozocrtx' in var_d[th_d_ind]['mat']) & ('vomecrty' in var_d[th_d_ind]['mat']):
            for ss in ['baroc_mag','baroc_phi', 'baroc_div', 'baroc_curl']: #,'dUdz','dVdz','abs_dUdz','abs_dVdz']: 
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 4
                    #var_grid[tmp_datstr][ss] = 'UV'
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['vozocrtx'] #'U'
                    var_d['d'].append(ss)

        if (('ubar' in var_d[th_d_ind]['mat']) & ('vbar' in var_d[th_d_ind]['mat'])) | (('vobtcrtx' in var_d[th_d_ind]['mat']) & ('vobtcrty' in var_d[th_d_ind]['mat'])) :
            for ss in ['barot_mag', 'barot_phi', 'barot_div', 'barot_curl']: 
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    if 'ubar' in var_d[th_d_ind]['mat']:
                        var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['ubar'] # 'U'
                    elif 'vobtcrtx' in var_d[th_d_ind]['mat']:
                        var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['vobtcrtx'] # 'U'
                    var_d['d'].append(ss)



        if ('vocetr_eff_e3v' in var_d[th_d_ind]['mat']) :
            for ss in ['StreamFunction_e3']: 
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    #var_grid[tmp_datstr][ss] = 'UV'
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['vocetr_eff_e3v'] # 'U'
                    var_d['d'].append(ss)


        if ('vocetr_eff' in var_d[th_d_ind]['mat']) :
            for ss in ['StreamFunction']: 
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    #var_grid[tmp_datstr][ss] = 'UV'
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['vocetr_eff'] # 'U'
                    var_d['d'].append(ss)


        if ('uocetr_eff_e3u' in var_d[th_d_ind]['mat']) & ('vocetr_eff_e3v' in var_d[th_d_ind]['mat']):
            for ss in ['VolTran_e3_mag', 'VolTran_e3_phi', 'VolTran_e3_div', 'VolTran_e3_curl']: 
                if ss not in var_d[th_d_ind]['mat']:
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    #var_grid[tmp_datstr][ss] = 'UV'
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['uocetr_eff_e3u'] # 'U'
                    var_d['d'].append(ss)



        if ('vocetr_eff' in var_d[th_d_ind]['mat']) & ('uocetr_eff' in var_d[th_d_ind]['mat']):
            for ss in ['VolTran_mag', 'VolTran_phi', 'VolTran_div', 'VolTran_curl']: 
                if ss not in var_d[th_d_ind]['mat']:
                    #ss = 'barot_mag'
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    #var_grid[tmp_datstr][ss] = 'UV'
                    var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['vocetr_eff'] #'U'
                    var_d['d'].append(ss)




        if ('uwnd' in var_d[th_d_ind]['mat']) & ('vwnd' in var_d[th_d_ind]['mat']):
            for ss in ['wnd_mag']:#, 'barot_div', 'barot_curl']: 
                if ss not in var_d[th_d_ind]['mat']:
                    #ss = 'barot_mag'
                    var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                    var_dim[ss] = 3
                    var_grid[tmp_datstr][ss] = 'WW3'
                    var_d['d'].append(ss)


        if ('N3n' in var_d[th_d_ind]['mat']) & ('N1p' in var_d[th_d_ind]['mat']):
            ss = 'N:P'
            if ss not in var_d[th_d_ind]['mat']:
                var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'],ss)
                var_dim[ss] = 4
                var_grid[tmp_datstr][ss] = var_grid[tmp_datstr]['N3n'] # 'T'
                var_d['d'].append(ss)


    return var_d,var_dim, var_grid




def regrid_2nd(regrid_params_in,regrid_meth,thd,configd,th_d_ind,dat_in): #):
    start_regrid_timer = datetime.now()

    # if its the same as the first config, don't regrid
    #     or if its the same before '_' (i.e. amm7 and amm7_LBC), don't regrid

    #
    # if configd[th_d_ind] == 'GULF18': pdb.set_trace()
    if (configd[th_d_ind].upper() == configd[1].upper())|(configd[th_d_ind].split('_')[0].upper() == configd[1].split('_')[0].upper())|((configd[th_d_ind].split('_')[0].upper() in ['AMM15','CO9P2','C09P2']) & (configd[1].split('_')[0].upper() in ['AMM15','CO9P2','C09P2'])):
        #print('dont regrid',configd[th_d_ind],configd[1])


        dat_out = dat_in
    else:
        (NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out) = regrid_params_in
        if (thd[1]['x0']!=0)|(thd[1]['y0']!=0): 
            print('thin_x0 and thin_y0 must equal 0, if not, need to work out thinning code in the regrid index method')
            #pdb.set_trace()
        #pdb.set_trace()

        if regrid_meth == 1:
            # Nearest Neighbour Interpolation   ~0.01 sec
            #dat_out = dat_in[NWS_amm_nn_jj_ind_final,NWS_amm_nn_ii_ind_final]
            #pdb.set_trace()
            #if NWS_amm_wgt_out.mask.sum(axis = 0).any(): pdb.set_trace()
            dat_out = dat_in[NWS_amm_nn_jj_ind_out,NWS_amm_nn_ii_ind_out]
            dat_out.mask = dat_out.mask|NWS_amm_wgt_out.mask.sum(axis =0)

        elif regrid_meth == 2:
            # Bilinear Interpolation            ~0.2sec

            dat_in_selected_corners =  dat_in[NWS_amm_bl_jj_ind_out ,NWS_amm_bl_ii_ind_out ].copy()
            NWS_amm_wgt_out.mask = NWS_amm_wgt_out.mask | dat_in_selected_corners.mask

            dat_out = (dat_in_selected_corners*NWS_amm_wgt_out).sum(axis = 0)/(NWS_amm_wgt_out).sum(axis = 0)
            

        else:
            print('config and configd_in must be AMM15 and AMM7')
            pdb.set_trace()
    
    #if verbose_debugging:  print ('Regrid timer for method #%i: '%regrid_meth, datetime.now() - start_regrid_timer)
    return dat_out



def grad_horiz_ns_data(thd,grid_dict,ii, iijj_ind, ns_slice_dict, meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False,
                       grad_horiz_vert_wgt = False,Sec_regrid_slice = False):
    Dataset_lst = [ss for ss in ns_slice_dict.keys()] 
    Dataset_lst.remove('x')
    Dataset_lst.remove('y')
    Dataset_lst.remove('Sec Grid')

    if Sec_regrid_slice == False:
        if meth == 0:
            ns_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e2t'][2:,ii] +  grid_dict['Dataset 1']['e2t'][:-2,ii])/2 + grid_dict['Dataset 1']['e2t'][1:-1,ii])
        elif meth == 1:
            ns_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e2t'][1:,ii] +  grid_dict['Dataset 1']['e2t'][:-1,ii])/2 )

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])


        if Sec_regrid_slice:
            tmp_dat = ns_slice_dict[tmp_datstr]


            if th_d_ind == 1:
                ii_in = ii
            else:
                ii_in = iijj_ind[tmp_datstr]['ii']
            
            if np.ma.is_masked(ii_in):
                continue
            



            if meth == 0:
                ns_slice_dx =  thd[th_d_ind]['dx']*((grid_dict[tmp_datstr]['e2t'][2:,ii_in] +  grid_dict[tmp_datstr]['e2t'][:-2,ii_in])/2 + grid_dict[tmp_datstr]['e2t'][1:-1,ii_in])
            elif meth == 1:
                ns_slice_dx =  thd[th_d_ind]['dx']*((grid_dict[tmp_datstr]['e2t'][1:,ii_in] +  grid_dict[tmp_datstr]['e2t'][:-1,ii_in])/2 )

            tmp_data_in = ns_slice_dict['Sec Grid'][tmp_datstr]['data']
        else:
            tmp_data_in = ns_slice_dict[tmp_datstr]


        if abs_pre:
            tmp_data_in = np.abs(tmp_data_in)
        else:
            tmp_data_in = tmp_data_in

        if meth == 0:
            dns_1 = tmp_data_in[:,2:] - tmp_data_in[:,:-2]

            if dx_d_dx:
                tmp_data_in[:,1:-1] = dns_1
            else:
                tmp_data_in[:,1:-1] = dns_1/ns_slice_dx

            tmp_data_in[:,-1] = np.ma.masked
        if meth == 1:
            dns_1 = tmp_data_in[:,1:] - tmp_data_in[:,:-1]

            if dx_d_dx:
                tmp_data_in[:,:-1] = dns_1
            else:
                tmp_data_in[:,:-1] = dns_1/ns_slice_dx
                
            tmp_data_in[:,-1] = np.ma.masked

        if abs_post:
            tmp_data_in = np.abs(tmp_data_in)

        if Sec_regrid_slice:
            ns_slice_dict['Sec Grid'][tmp_datstr]['data'] = tmp_data_in
        else:
            ns_slice_dict[tmp_datstr] = tmp_data_in


    return ns_slice_dict


def grad_horiz_ew_data(thd,grid_dict,jj, iijj_ind, ew_slice_dict, meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False,
                       grad_horiz_vert_wgt = False,Sec_regrid_slice = False):
    Dataset_lst = [ss for ss in ew_slice_dict.keys()] 
    Dataset_lst.remove('x')
    Dataset_lst.remove('y')
    Dataset_lst.remove('Sec Grid')


    if Sec_regrid_slice == False:
        if meth == 0:
            ew_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e1t'][jj,2:] +  grid_dict['Dataset 1']['e1t'][jj,:-2])/2 + grid_dict['Dataset 1']['e1t'][jj,1:-1])
        elif meth == 1:
            ew_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e1t'][jj,1:] +  grid_dict['Dataset 1']['e1t'][jj,:-1])/2)

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        if Sec_regrid_slice:
            if th_d_ind == 1:
                jj_in = jj
            else:
                jj_in = iijj_ind[tmp_datstr]['jj']



            if np.ma.is_masked(jj_in):
                continue

            if meth == 0:
                ew_slice_dx =  thd[th_d_ind]['dx']*((grid_dict[tmp_datstr]['e1t'][jj_in,2:] +  grid_dict[tmp_datstr]['e1t'][jj_in,:-2])/2 + grid_dict[tmp_datstr]['e1t'][jj_in,1:-1])
            elif meth == 1:
                ew_slice_dx =  thd[th_d_ind]['dx']*((grid_dict[tmp_datstr]['e1t'][jj_in,1:] +  grid_dict[tmp_datstr]['e1t'][jj_in,:-1])/2)

            tmp_data_in = ew_slice_dict['Sec Grid'][tmp_datstr]['data']
        else:
            tmp_data_in = ew_slice_dict[tmp_datstr]


        if abs_pre:
            tmp_data_in = np.abs(tmp_data_in)
        else:
            tmp_data_in = tmp_data_in

        if meth == 0:
            dew_1 = tmp_data_in[:,2:] - tmp_data_in[:,:-2]

            if dx_d_dx:
                tmp_data_in[:,1:-1] = dew_1
            else:
                tmp_data_in[:,1:-1] = dew_1/ew_slice_dx
                
            tmp_data_in[:,-1] = np.ma.masked
        elif meth == 1:
            dew_1 = tmp_data_in[:,1:] - tmp_data_in[:,:-1]

            if dx_d_dx:
                tmp_data_in[:,:-1] = dew_1
            else:
                tmp_data_in[:,:-1] = dew_1/ew_slice_dx
            tmp_data_in[:,-1] = np.ma.masked

        if abs_post:
            tmp_data_in = np.abs(tmp_data_in)

        if Sec_regrid_slice:
            ew_slice_dict['Sec Grid'][tmp_datstr]['data'] = tmp_data_in
        else:
            ew_slice_dict[tmp_datstr] = tmp_data_in

        
    return ew_slice_dict


def grad_vert_ns_data(ns_slice_dict, meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False,Sec_regrid_slice = False):
    
    # meth = 0 is centred differnce, 1 = forward diff

    
    Dataset_lst = [ss for ss in ns_slice_dict.keys()] 
    Dataset_lst.remove('x')
    Dataset_lst.remove('y')
    Dataset_lst.remove('Sec Grid')
    ns_slice_y = ns_slice_dict['y']
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])



        if Sec_regrid_slice:
            ns_slice_y = ns_slice_dict['Sec Grid'][tmp_datstr]['y'].copy()
            if meth == 0:
                dns_z = ns_slice_y[2:,:] - ns_slice_y[:-2,:]
            elif meth == 1:
                dns_z = ns_slice_y[1:,:] - ns_slice_y[:-1,:]

            tmp_data_in =  ns_slice_dict['Sec Grid'][tmp_datstr]['data'].copy()
        else:
            tmp_data_in = ns_slice_dict[tmp_datstr].copy()


        if meth == 0:
            dns_z = ns_slice_y[2:,:] - ns_slice_y[:-2,:]
        elif meth == 1:
            dns_z = ns_slice_y[1:,:] - ns_slice_y[:-1,:]


        if abs_pre:
            tmp_data_in = np.abs(tmp_data_in)


        if meth == 0:
            dns_1 = tmp_data_in[2:,:] - tmp_data_in[:-2,:]

            if dx_d_dx:
                tmp_data_in[1:-1,:] = dns_1
            else:
                tmp_data_in[1:-1,:] = dns_1/dns_z

            tmp_data_in[ 0,:] = np.ma.masked
            tmp_data_in[-1,:] = np.ma.masked
        elif meth == 1:
            dns_1 = tmp_data_in[1:,:] - tmp_data_in[:-1,:]

            if dx_d_dx:
                tmp_data_in[:-1,:] = dns_1
            else:
                tmp_data_in[:-1,:] = dns_1/dns_z

            tmp_data_in[-1,:] = np.ma.masked

        if abs_post:
            tmp_data_in= np.abs(tmp_data_in)

        if Sec_regrid_slice:
            ns_slice_dict['Sec Grid'][tmp_datstr]['data'] = tmp_data_in
        else:
            ns_slice_dict[tmp_datstr] = tmp_data_in

    return ns_slice_dict


def grad_vert_ew_data(ew_slice_dict, meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False, Sec_regrid_slice = False):
    
    # meth = 0 is centred differnce, 1 = forward diff


    Dataset_lst = [ss for ss in ew_slice_dict.keys()] 
    Dataset_lst.remove('x')
    Dataset_lst.remove('y')
    Dataset_lst.remove('Sec Grid')

    ew_slice_y = ew_slice_dict['y']

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])




        if Sec_regrid_slice:
            ew_slice_y = ew_slice_dict['Sec Grid'][tmp_datstr]['y'].copy()
            if meth == 0:
                dew_1 = ew_slice_y[2:,:] - ew_slice_y[:-2,:]
            elif meth == 1:
                dew_1 = ew_slice_y[1:,:] - ew_slice_y[:-1,:]

            tmp_data_in =  ew_slice_dict['Sec Grid'][tmp_datstr]['data'].copy()
        else:
            tmp_data_in = ew_slice_dict[tmp_datstr].copy()


        if meth == 0:
            dew_z = ew_slice_y[2:,:] - ew_slice_y[:-2,:]
        elif meth == 1:
            dew_z = ew_slice_y[1:,:] - ew_slice_y[:-1,:]


        if abs_pre:
            tmp_data_in = np.abs(tmp_data_in)

            
        if meth == 0:
            dew_1 = tmp_data_in[2:,:] - tmp_data_in[:-2,:]
            if dx_d_dx:
                tmp_data_in[1:-1,:] = dew_1
            else:
                tmp_data_in[1:-1,:] = dew_1/dew_z

            tmp_data_in[ 0,:] = np.ma.masked
            tmp_data_in[-1,:] = np.ma.masked
        elif meth == 1:
            dew_1 = tmp_data_in[1:,:] - tmp_data_in[:-1,:]
            if dx_d_dx:
                tmp_data_in[:-1,:] = dew_1
            else:
                tmp_data_in[:-1,:] = dew_1/dew_z

            tmp_data_in[-1,:] = np.ma.masked


        if abs_post:
            tmp_data_in= np.abs(tmp_data_in)

        if Sec_regrid_slice:
            ew_slice_dict['Sec Grid'][tmp_datstr]['data'] = tmp_data_in
        else:
            ew_slice_dict[tmp_datstr] = tmp_data_in
    return ew_slice_dict




def grad_vert_hov_prof_data(hov_dat_dict, meth=0, abs_pre = False, abs_post = False, regrid_xy = False,dx_d_dx = False):

  

    # meth = 0 is centred differnce, 1 = forward diff

    Dataset_lst = [ss for ss in hov_dat_dict.keys()]   
    if 'x' in Dataset_lst: Dataset_lst.remove('x')       
    if 'y' in Dataset_lst: Dataset_lst.remove('y')    
    
    if 'Sec Grid' in Dataset_lst: Dataset_lst.remove('Sec Grid')    


    hov_y = hov_dat_dict['y']
    if meth == 0:
        dhov_z = hov_y[2:] - hov_y[:-2]
    elif meth == 1:
        dhov_z = hov_y[1:] - hov_y[:-1]
    
    for tmp_datstr in Dataset_lst:

        if abs_pre:
            tmp_data_in = np.abs(hov_dat_dict[tmp_datstr])
        else:
            tmp_data_in = hov_dat_dict[tmp_datstr]

        if meth == 0:
            dhov_1 = tmp_data_in[2:] - tmp_data_in[:-2]
            if dx_d_dx:
                hov_dat_dict[tmp_datstr][1:-1] = (dhov_1.T).T
            else:
                hov_dat_dict[tmp_datstr][1:-1] = (dhov_1.T/dhov_z).T
            hov_dat_dict[tmp_datstr][ 0] = np.ma.masked
            hov_dat_dict[tmp_datstr][-1] = np.ma.masked
        elif meth == 1:
            dhov_1 = tmp_data_in[1:] - tmp_data_in[:-1]
            if dx_d_dx:
                hov_dat_dict[tmp_datstr][:-1] = (dhov_1.T).T
            else:
                hov_dat_dict[tmp_datstr][:-1] = (dhov_1.T/dhov_z).T
            hov_dat_dict[tmp_datstr][-1] = np.ma.masked

        hov_dat_dict[tmp_datstr][ 0] = np.ma.masked
        hov_dat_dict[tmp_datstr][-1] = np.ma.masked


        if abs_post:
            hov_dat_dict[tmp_datstr] = np.abs(hov_dat_dict[tmp_datstr])


    return hov_dat_dict


def connect_to_files_with_xarray(Dataset_lst,fname_dict,xarr_dict,nldi,ldi_ind_mat, ld_lab_mat,ld_nctvar,
    force_dim_d = None,xarr_rename_master_dict=None,gr_1st = 'T',do_addtimedim = None, do_all_WW3 = False):
    # connect to files with xarray, and create dictionaries with vars, dims, grids, time etc. 

    do_addtimedim = True

    WW3_var_lst = ['hs','tp','t0m1','dp','spr','uwnd','vwnd','ucur','vcur',]
    # NB xarr_dict is not passed back.
    
    import xarray

    #from dask.distributed import Client
    #client = Client(n_workers=20, threads_per_worker=2, memory_limit='7.5GB')

    var_d = {}
    var_d['d'] = []
    var_dim = {}
    var_grid = {}
    ncvar_d = {}
    ncdim_d = {}
    time_d = {}
    WW3_ld_nctvar = 'time'
    init_timer = []
    # open file list with xarray
    #xarr_dict = {}

    
    init_timer.append((datetime.now(),'xarray open_mfdataset connecting'))
    print ('xarray open_mfdataset connecting',datetime.now())

    for tmp_datstr in Dataset_lst: # xarr_dict.keys():
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        
        var_d[th_d_ind] = {}
        var_d[th_d_ind]['mat'] = np.array([])
        var_grid[tmp_datstr] = {}
        ncvar_d[tmp_datstr] = {}
        ncdim_d[tmp_datstr] = {}
        time_d[tmp_datstr] = {}
        #xarr_dict[tmp_datstr] = {}
        for tmpgrid in xarr_dict[tmp_datstr].keys():
            #pdb.set_trace()
            time_d[tmp_datstr][tmpgrid] = {}
            #pdb.set_trace()
            # Increments don't have muliple lead times like other grids.
            #if tmpgrid != 'I':
            if tmpgrid not in  ['I','In','Ic']: # 
                #pdb.set_trace()
                if (nldi == 0) : # If only loading one lead time:   
                    #xarr_dict[tmp_datstr][tmpgrid].append(
                    #    xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                    #    combine='by_coords',parallel = True))  
                    '''
                    '''
                    ###WW3 time ncvar name
                    # may be required if a problem with WW3 having 25 instantaneous hours every 25 hours.
                    # if loading in many daily files, there will be repeated hours, so, only load the last 24 hours of the files. 
                    if tmpgrid == 'WW3':
                        xarr_dict[tmp_datstr][tmpgrid].append(
                            xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                            combine='by_coords',parallel = True, preprocess=lambda ds: ds[{WW3_ld_nctvar:slice(1,24+1)}]))    
                    else:
                        xarr_dict[tmp_datstr][tmpgrid].append(
                            xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                            combine='by_coords',parallel = True)) 
                        #pdb.set_trace()
                        
                else: # if loading different lead times. 
                    #pdb.set_trace()
                    '''
                    fc_nday_ldtime = 2
                    
                    
                    tmp_xarr_data = xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid],combine='nested', concat_dim='time_counter', parallel = True)
                    tmptc_np64 = tmp_xarr_data.time_counter.load()
                    tmptc = np.array([datetime.utcfromtimestamp(((ss.data -  np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1, 's'))) for ss in tmptc_np64])

                    #fc_len = np.diff(np.where(np.array([ss.days for ss in np.diff(tmptc)])!=1)[0])
                    fc_len_mat = np.diff(np.where(np.array([ss.days for ss in np.diff(tmptc)])<1)[0])
                    
                    if len(fc_len_mat) == 0:
                        fc_len = 1
                    else:
                        if fc_len_mat.std() != 0:
                            print('not all fc are the same length',fc_len_mat)
                            pdb.set_trace()
                        fc_len = ((fc_len_mat).mean()).astype('int')
                    
                    
                    bltc = tmptc[2::fc_len]
                    bltc_mat = np.array([ss  for ss in bltc for i_i in range(8)] )
                    ldtc_mat = np.array([ss.days for ss in (tmptc - bltc_mat)])


                    tmp_xarr_data = tmp_xarr_data.assign_coords(bull_time = bltc_mat)
                    tmp_xarr_data = tmp_xarr_data.assign_coords(lead_time = ldtc_mat)
               
                    tmpgpby_bull = tmp_xarr_data.groupby('bull_time')
                    tmpgpby_lead = tmp_xarr_data.groupby('lead_time')

                    for ss in tmpgpby_bull.groups: ss,tmpgpby_bull.groups[ss]
                    for ss in tmpgpby_lead.groups: ss,tmpgpby_lead.groups[ss]

                    tmp_xarr_data.votemper[tmpgpby_lead.groups[-2],:,:,:]
                    tmp_xarr_data.votemper[tmpgpby_bull.groups[np.datetime64('2024-05-11T12:00:00.000000000')],:,:,:]


                    rootgrp_hpc_time = Dataset('/scratch/frwave/wave_rolling_archive/amm15/amm15_2024101200.nc', 'r', format='NETCDF4')
                    fcper = rootgrp_hpc_time.variables['forecast_period'][:]
                    rootgrp_hpc_time.close()

                    pdb.set_trace()





                    -12,  fcper[:24]/86400
                    12, fcper[25:25+24]/86400
                    36, fcper[25+24:25+24+24]/86400
                    60, fcper[25+24+24:25+24+24+24]/86400
                    84, fcper[25+72+1:25+72+1+24]/86400
                    108, fcper[25+72+1+24:25+72+1+24+24]/86400
                    132, fcper[25+72+1+24+24:25+72+1+24+24+24]/86400
                    156, fcper[25+72+1+24+24+24:25+72+1+24+24+24+24]/86400


                    -12,  fcper[1:24+1]/86400
                    12, fcper[25+1:25+24+1]/86400
                    36, fcper[25+1+24:25+24+24+1]/86400
                    60, fcper[25+1+24+24:25+24+24+24+1]/86400
                    84, fcper[25+1+72+1:25+72+1+24+1]/86400
                    108, fcper[25+1+72+1+24:25+72+1+24+24+1]/86400
                    132, fcper[25+1+72+1+24+24:25+72+1+24+24+24+1]/86400

                    -12,1,24+1
                    12,25+1,25+24+1
                    36,25+1+24,25+24+24+1
                    60, 25+1+24+24,25+24+24+24+1
                    84, 25+1+72+1,25+72+1+24+1
                    108,   25+1+72+1+24,25+72+1+24+24+1
                    132,   25+1+72+1+24+24,25+72+1+24+24+24+1




                    '''
                    ###WW3 time ncvar name
                    for li,(ldi,ldilab) in enumerate(zip(ldi_ind_mat, ld_lab_mat)): 
                        if tmpgrid == 'WW3':
                            #xarr_dict[tmp_datstr][tmpgrid].append(
                            #xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                            #combine='by_coords',parallel = True, preprocess=lambda ds: ds[{WW3_ld_nctvar:slice(1,24+1)}]))    
                            if ldilab in ['-12','-36']:
                                WW3_ldi0, WW3_ldi1 = 1,25
                            elif ldilab  == '012':
                                WW3_ldi0, WW3_ldi1 = 26,50
                            elif ldilab  == '036':
                                WW3_ldi0, WW3_ldi1 = 50,74
                            elif ldilab  == '060':
                                WW3_ldi0, WW3_ldi1 = 74,98
                            elif ldilab  == '084':
                                WW3_ldi0, WW3_ldi1 = 99,123
                            elif ldilab  == '108':
                                WW3_ldi0, WW3_ldi1 = 123,147
                            elif ldilab  == '132':
                                WW3_ldi0, WW3_ldi1 = 147,171
                            else:
                                pdb.set_trace()
                                WW3_ldi0, WW3_ldi1 = 1,25

                            xarr_dict[tmp_datstr][tmpgrid].append(
                            xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                            combine='by_coords',parallel = True, preprocess=lambda ds: ds[{WW3_ld_nctvar:slice(WW3_ldi0, WW3_ldi1)}]))    
                        else:
                            #print(li,(ldi,ldilab))
                            xarr_dict[tmp_datstr][tmpgrid].append(
                                xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                                combine='by_coords',parallel = True, preprocess=lambda ds: ds[{ld_nctvar:slice(ldi,ldi+1)}]))   

                    '''
                    for li,(ldi,ldilab) in enumerate(zip(ldi_ind_mat, ld_lab_mat)): 
                        #print(li,(ldi,ldilab))
                        xarr_dict[tmp_datstr][tmpgrid].append(
                            xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid], 
                            combine='by_coords',parallel = True, preprocess=lambda ds: ds[{ld_nctvar:slice(ldi,ldi+1)}]))   
                    '''
            #elif tmpgrid == 'I':
            else: # elif tmpgrid in  ['I','In','Ic']: # 
                #pdb.set_trace()
                #tmp_T_time_datetime,tmp_T_time_datetime_since_1970,ntime,ti, nctime_calendar_type = extract_time_from_xarr(xarr_dict['Dataset 1']['T'],fname_dict['Dataset 1']['T'][0],'time_counter','time_counter',None,'%Y%m%d',1,False)
                tmp_T_time_datetime,tmp_T_time_datetime_since_1970,ntime,ti, nctime_calendar_type = extract_time_from_xarr(xarr_dict['Dataset 1'][gr_1st],fname_dict['Dataset 1'][gr_1st][0],'time_counter','time_counter',None,'%Y%m%d',1,False)
                if tmp_T_time_datetime.size == len(fname_dict[tmp_datstr][tmpgrid]):
                    inc_T_time_datetime = tmp_T_time_datetime
                    inc_T_time_datetime_since_1970 = tmp_T_time_datetime_since_1970
                else:
                    inc_T_time_datetime = [tmp_T_time_datetime[0] + timedelta(days = i_i) for i_i in range(len(fname_dict[tmp_datstr][tmpgrid]))]
                    tmp_T_time_datetime_since_1970 = [tmp_T_time_datetime_since_1970[0] + i_i*86400 for i_i in range(len(fname_dict[tmp_datstr][tmpgrid]))]
                
                time_d[tmp_datstr][tmpgrid]['datetime'] = np.array(inc_T_time_datetime)
                time_d[tmp_datstr][tmpgrid]['datetime_since_1970'] = np.array(tmp_T_time_datetime_since_1970)

                tmp_xarr_data = xarray.open_mfdataset(fname_dict[tmp_datstr][tmpgrid],combine='nested', concat_dim='t', parallel = True)
                tmp_xarr_data.assign_coords(t = time_d[tmp_datstr][tmpgrid]['datetime'])
                xarr_dict[tmp_datstr][tmpgrid].append(tmp_xarr_data)

            

            if do_addtimedim:
                for tmpldi in range(len(xarr_dict[tmp_datstr][tmpgrid])):
                    # You could be comparing a file with a time dimension with one that doesn't have one,
                    # so check if this dataset has a time dimension - don't add if already there
                    poss_tdims = ['time_counter','time','t']
                    tmp_addtimedim = True
                    for ss in xarr_dict[tmp_datstr][tmpgrid][tmpldi].dims.keys():
                        if ss.lower() in poss_tdims:
                            tmp_addtimedim = False
                    if tmp_addtimedim:
                        #pdb.set_trace()
                        xarr_dict[tmp_datstr][tmpgrid][tmpldi] = xarr_dict[tmp_datstr][tmpgrid][tmpldi].expand_dims(dim={"time_counter": 1})
                        #pdb.set_trace()
                        xarr_dict[tmp_datstr][tmpgrid][tmpldi]["time_counter"]=(['time_counter'],  [0.])
                        xarr_dict[tmp_datstr][tmpgrid][tmpldi]["time_counter"].attrs = {'standard_name':"time",'long_name':"Time axis",'calendar':"gregorian",'units':"seconds since  2025-01-01 00:00:00",'time_origin':" 2025-01-01 00:00:00"}



                    '''
                    time_counter:least_significant_digit = 4 ;
                    time_counter:axis = "T" ;
                    time_counter:standard_name = "time" ;
                    time_counter:long_name = "Time axis" ;
                    time_counter:calendar = "gregorian" ;
                    time_counter:units = "seconds since  2010-01-01 00:00:00" ;
                    time_counter:time_origin = " 2010-01-01 00:00:00" ;
                    time_counter:bounds = "time_counter_bounds" ;


                    axis = "T" ;
                    
                    '''

                    #xarr_dict[tmp_datstr][tmpgrid][tmpldi].["time_counter"] = tmpnctime

                    #xarr_dict[tmp_datstr][tmpgrid][tmpldi].Dataset({"time_counter": [0]})
                    #xarr_dict[tmp_datstr][tmpgrid][tmpldi].assign_attrs(units="Celsius", description="Temperature data

            
                #.expand_dims(dim={"t": 1})
            #pdb.set_trace()
            
            # When comparing files/models, only variables that are common to both Datasets are shown. 
            # If comparing models with different names for the same variables, they won't be shown, 
            # as temperature and votemper will be considered different.
            #
            # We can use xarray to rename the variables as they are loaded to overcome this, using a rename_dictionary.
            # i.e. rename any variables called tmperature or temp to votemper etc.
            # to do this, we use the following command line arguments:
            # --rename_var votemper temperature temp --rename_var vosaline salinity sal 
            # where each variable has its own instance, and the first entry is what it will be renamed too, 
            # and the remaining entries are renamed. 

            #xarr_rename_master_dict = None
            if xarr_rename_master_dict is not None:
                
                xarr_rename_dict = {}
                tmp_cur_var = [ss for ss in xarr_dict[tmp_datstr][tmpgrid][-1].variables.keys() ]
                for ss in tmp_cur_var: 
                    if ss in xarr_rename_master_dict.keys():
                        xarr_rename_dict[ss] = xarr_rename_master_dict[ss]
                        #xarr_rename_dict[xarr_rename_master_dict[ss]] = ss
                #pdb.set_trace()
                if len(xarr_rename_dict)>0:
                    xarr_dict[tmp_datstr][tmpgrid][-1] = xarr_dict[tmp_datstr][tmpgrid][-1].rename(xarr_rename_dict)

            
            init_timer.append((datetime.now(),'xarray open_mfdataset %s %s connected'%(tmp_datstr,tmpgrid)))
            print ('xarray open_mfdataset %s %s, Loaded'%(tmp_datstr,tmpgrid),datetime.now())
            #pdb.set_trace()
            ncvar_mat = [ss for ss in xarr_dict[tmp_datstr][tmpgrid][0].variables.keys()]
            
    
            ncvar_d[tmp_datstr][tmpgrid] = ncvar_mat
            tmp_x_dim, tmp_y_dim, tmp_z_dim, tmp_t_dim  = load_nc_dims(xarr_dict[tmp_datstr][tmpgrid][0]) #  find the names of the x, y, z and t dimensions.
            #pdb.set_trace()


            # If files have more than grid, with differing dimension for each, you can enforce the dimenson for each grid.
            # For example, the SMHI BAL-MFC NRT system (BALMFCorig) hourly surface files hvae the T, U, V and T_inner grid in the same file. 
            # Load the smae file in for each grid:
            # Nslvdev BALMFCorig NS01_SURF_2025020912_1-24H.nc 
            # --files 1 U NS01_SURF_2025020912_1-24H.nc --files 1 V NS01_SURF_2025020912_1-24H.nc --files 1 Ti NS01_SURF_2025020912_1-24H.nc 
            # .....   --th 1 dxy 
            # and enforce which dimensions are used for the T, U, V and T_inner grid (x,y,z and T)
            #--forced_dim U x x_grid_U y y_grid_U --forced_dim V x x_grid_V y y_grid_V --forced_dim T x x_grid_T y y_grid_T --forced_dim Ti x x_grid_T_inner y y_grid_T_inner
            #
            #
            # force_dim_d_in = 
            # {'U': {'x': 'x_grid_U', 'y': 'y_grid_U'}, 'V': {'x': 'x_grid_V', 'y': 'y_grid_V'}, 'T': {'x': 'x_grid_T', 'y': 'y_grid_T'}, 'x': {'x_grid_T_inner': 'y'}}

            if force_dim_d is not None:
                #pdb.set_trace()
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                if th_d_ind in force_dim_d.keys():
                    if tmpgrid in force_dim_d[th_d_ind].keys():
                        if 'x' in force_dim_d[th_d_ind][tmpgrid].keys(): tmp_x_dim = force_dim_d[th_d_ind][tmpgrid]['x']
                        if 'y' in force_dim_d[th_d_ind][tmpgrid].keys(): tmp_y_dim = force_dim_d[th_d_ind][tmpgrid]['y']
                        if 'z' in force_dim_d[th_d_ind][tmpgrid].keys(): tmp_z_dim = force_dim_d[th_d_ind][tmpgrid]['z']
                        if 't' in force_dim_d[th_d_ind][tmpgrid].keys(): tmp_t_dim = force_dim_d[th_d_ind][tmpgrid]['t']
            
            tmp_var_names = load_nc_var_name_list(xarr_dict[tmp_datstr][tmpgrid][0], tmp_x_dim, tmp_y_dim, tmp_z_dim,tmp_t_dim)# find the variable names in the nc file # var_4d_mat, var_3d_mat, var_d[1]['T'], nvar4d, nvar3d, nvar, var_dim = 
            ncdim_d[tmp_datstr][tmpgrid]  = {}
            ncdim_d[tmp_datstr][tmpgrid]['t'] = tmp_t_dim
            ncdim_d[tmp_datstr][tmpgrid]['z'] = tmp_z_dim
            ncdim_d[tmp_datstr][tmpgrid]['y'] = tmp_y_dim
            ncdim_d[tmp_datstr][tmpgrid]['x'] = tmp_x_dim
        
            tmp_var_dim = tmp_var_names[6]
            var_d[th_d_ind][tmpgrid] = tmp_var_names[2]

            if tmpgrid == 'WW3':
                tmp_WW3_var_mat,  WW3_nvar, tmp_var_dim = load_nc_var_name_list_WW3(xarr_dict[tmp_datstr][tmpgrid][0],'seapoint',tmp_t_dim)
                if do_all_WW3:
                    WW3_var_mat = [ss for ss in tmp_WW3_var_mat]
                else:
                    WW3_var_mat = [ss for ss in tmp_WW3_var_mat if ss in WW3_var_lst]
                var_d[th_d_ind][tmpgrid] = WW3_var_mat
                #pdb.set_trace()


            
            for ss in tmp_var_dim: var_dim[ss] = tmp_var_dim[ss]


            var_d[th_d_ind]['mat'] = np.append(var_d[th_d_ind]['mat'] , var_d[th_d_ind][tmpgrid])
    
    
    #pdb.set_trace()
    '''
    for ss in var_d[1]:
        for tmpgrid in var_d[1].keys():
            for ss in var_d[1][tmpgrid]: var_grid['Dataset 1'][ss] = tmpgrid
    '''
    for tmp_datstr in Dataset_lst: # xarr_dict.keys():
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        
        #for ss in var_d[th_d_ind]:  #['mat', 'T', 'U', 'V']  or ['mat', 'T', 'T_bt_1', 'U_bt_1', 'V_bt_1']
        #pdb.set_trace()
        for tmpgrid in var_d[th_d_ind].keys():
            for ss in var_d[th_d_ind][tmpgrid]: 

                #var_grid[tmp_datstr][ss] = tmpgrid


                if tmpgrid == 'mat': continue
                if ss in var_grid[tmp_datstr].keys():
                    var_grid[tmp_datstr][ss].append(tmpgrid)
                else:    
                    var_grid[tmp_datstr][ss] = [tmpgrid]

    
    #pdb.set_trace()

        
    #pdb.set_trace()
    return var_d,var_dim,var_grid,ncvar_d,ncdim_d,time_d


def remove_extra_end_file_dict(fname_dict):
    # For a given grid, if some datasets have more files than others, 
    # remove the additional end files from the longer dataset.
    # Often the operational system has only run AMM7 before AMM15, 
    # so there are more amm7 files avaialble. this causes issues.

    for tmpgrid in fname_dict['Dataset 1'].keys():
        nfliles_per_grid_lst = []
        for tmp_datstr in fname_dict.keys():
            nfliles_per_grid_lst.append(len(fname_dict[tmp_datstr][tmpgrid]))
        nfliles_per_grid_mat = np.array(nfliles_per_grid_lst)

        #if nfliles_per_grid_mat.ptp()>0:
        if np.ptp(nfliles_per_grid_mat)>0:
            print('\n\nDiffering number of files between dataset, for grid %s.\nRemoving extra end files.\n\nPress c to continue'%tmpgrid)
            pdb.set_trace()
            first_nfiles = nfliles_per_grid_mat.min()
            for tmp_datstr in fname_dict.keys():
                fname_dict[tmp_datstr][tmpgrid] = fname_dict[tmp_datstr][tmpgrid][:first_nfiles]


    return fname_dict
def trim_file_dict(fname_dict,thd):
    #pdb.set_trace()
    for tmp_datstr in fname_dict.keys():
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        for tmpgrid in fname_dict[tmp_datstr].keys():
            #pdb.set_trace()
            fname_dict[tmp_datstr][tmpgrid] = fname_dict[tmp_datstr][tmpgrid][  thd[th_d_ind]['f0']:thd[th_d_ind]['f1']:thd[th_d_ind]['df']]
    return fname_dict


def create_col_lst(nDataset):
    #create a set of lists of standard colours, colours for differences, and linestyles.
    
    Dataset_col = ['r','b','g','c','m','y']
    Dataset_col_diff = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    #https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/linestyles.html
    linestyle_str = [ 'solid', 'dotted','dashed','dashdot','loosely dotted',  (0, (1, 10)), (0, (5, 10)),   (0, (5, 1)), (0, (3, 10, 1, 10)) , (0, (3, 1, 1, 1)) , (0, (3, 5, 1, 5, 1, 5))  ,(0, (3, 10, 1, 10, 1, 10))  , (0, (3, 1, 1, 1, 1, 1))  ]
    
    # if need more 
    if nDataset>len(Dataset_col):
        import matplotlib.colors as mcolors
        CSS4_COLORS = np.array([ss for ss in mcolors.CSS4_COLORS.keys()])
        
        for ii in range(nDataset-len(Dataset_col)):Dataset_col.append(CSS4_COLORS[ii])

    if (nDataset**2)>len(Dataset_col):
        import matplotlib.colors as mcolors
        XKCD_COLORS = np.array([ss for ss in mcolors.XKCD_COLORS.keys()])
        for ii in range((nDataset**2)-len(Dataset_col_diff)):Dataset_col_diff.append(XKCD_COLORS[ii])
    #print (nDataset,len(Dataset_col),len(Dataset_col_diff),len(linestyle_str))
    return Dataset_col,Dataset_col_diff,linestyle_str

def create_xarr_dict(fname_dict):
    #create an empty dictionary with correct datasets and grids

   
    xarr_dict = {}
    for tmp_datstr in fname_dict.keys():
        xarr_dict[tmp_datstr] = {}
        for tmpgrid in fname_dict[tmp_datstr].keys():
            xarr_dict[tmp_datstr][tmpgrid] = []

    return xarr_dict


def create_Dataset_lst(fname_dict):


    Dataset_lst = []

    for tmp_datstr in fname_dict.keys():
        Dataset_lst.append(tmp_datstr)
    nDataset = len(Dataset_lst)


    return Dataset_lst,nDataset





#def load_grid_dict(Dataset_lst,rootgrp_gdept_dict, thd, nce1t,nce2t,nce3t,configd, config_fnames_dict,cutxind,cutyind,cutout_data, do_mask_dict):
def load_grid_dict(Dataset_lst, rootgrp_gdept_dict, thd, nce1t,nce2t,nce3t,configd, config_fnames_dict, cutout_d, do_mask_dict):
    grid_dict = {}
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

        do_mask = do_mask_dict[tmp_datstr]


        cutxind,cutyind,cutout_data = cutout_d[th_d_ind]['cutxind'],cutout_d[th_d_ind]['cutyind'],cutout_d[th_d_ind]['do_cutout']

        grid_dict[tmp_datstr] = {}
        if cutout_data:
            #pdb.set_trace()
            grid_dict[tmp_datstr]['e1t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce1t][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            grid_dict[tmp_datstr]['e2t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce2t][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            grid_dict[tmp_datstr]['e3t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce3t][0,:,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            tmp_configd = configd[th_d_ind]
            if tmp_configd is None:tmp_configd = configd[1]
            grid_dict[tmp_datstr]['gdept'] = rootgrp_gdept_dict[tmp_datstr].variables[config_fnames_dict[tmp_configd]['ncgdept']][0,:,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            if do_mask:
                grid_dict[tmp_datstr]['tmask'] = rootgrp_gdept_dict[tmp_datstr].variables['tmask'][0,:,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
                
        else:
            grid_dict[tmp_datstr]['e1t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce1t][0,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            grid_dict[tmp_datstr]['e2t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce2t][0,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            grid_dict[tmp_datstr]['e3t'] = rootgrp_gdept_dict[tmp_datstr].variables[nce3t][0,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            tmp_configd = configd[th_d_ind]
            if tmp_configd is None:tmp_configd = configd[1]
            grid_dict[tmp_datstr]['gdept'] = rootgrp_gdept_dict[tmp_datstr].variables[config_fnames_dict[tmp_configd]['ncgdept']][0,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            if do_mask:
                grid_dict[tmp_datstr]['tmask'] = rootgrp_gdept_dict[tmp_datstr].variables['tmask'][0,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']]
            
    nz = grid_dict[Dataset_lst[0]]['gdept'].shape[0]



    return grid_dict,nz



def create_config_fnames_dict(configd,Dataset_lst,script_dir):
    
    # create a dictionary with all the config info in it

    config_fnames_dict = {}
    #config_fnames_dict[configd[1]] = {}

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        config_fnames_dict[configd[th_d_ind]] = {}
        config_csv_fname = script_dir + 'NEMO_nc_slevel_viewer_config_%s.csv'%configd[th_d_ind].upper()
        with open(config_csv_fname, mode='r') as infile:           
            reader = csv.reader(infile)
            #pdb.set_trace()
            for rows in reader :config_fnames_dict[configd[th_d_ind]][rows[0]] = rows[1]


    return config_fnames_dict

def create_rootgrp_gdept_dict(config_fnames_dict,Dataset_lst,configd,use_xarray_gdept = False):
    # create dictionary with mesh files handles

    #rootgrp_gdept = None
    rootgrp_gdept_dict = {}

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        if use_xarray_gdept:
            rootgrp_gdept_dict[tmp_datstr] = xarray.open_dataset(config_fnames_dict[configd[th_d_ind]]['mesh_file'])
        else:
            rootgrp_gdept_dict[tmp_datstr] = Dataset(config_fnames_dict[configd[th_d_ind]]['mesh_file'], 'r', format='NETCDF4')


    return rootgrp_gdept_dict


def create_gdept_ncvarnames(config_fnames_dict,configd):
    #set ncvariable names for mesh files from info in config file, via config_fnames_dict
    ncgdept = 'gdept_0'
    nce1t = 'e1t'
    nce2t = 'e2t'
    nce3t = 'e3t_0'
    ncglamt = 'glamt'
    ncgphit = 'gphit'

    if 'ncgdept' in config_fnames_dict[configd[1]].keys():ncgdept = config_fnames_dict[configd[1]]['ncgdept']
    if 'nce1t' in config_fnames_dict[configd[1]].keys():nce1t = config_fnames_dict[configd[1]]['nce1t']
    if 'nce2t' in config_fnames_dict[configd[1]].keys():nce2t = config_fnames_dict[configd[1]]['nce2t']
    if 'nce3t' in config_fnames_dict[configd[1]].keys():nce3t = config_fnames_dict[configd[1]]['nce3t']
    if 'ncglamt' in config_fnames_dict[configd[1]].keys():ncglamt = config_fnames_dict[configd[1]]['ncglamt']
    if 'ncgphit' in config_fnames_dict[configd[1]].keys():ncgphit = config_fnames_dict[configd[1]]['ncgphit']

    return ncgdept,nce1t,nce2t,nce3t,ncglamt,ncgphit



def create_ncvar_lon_lat_time_dict(ncvar_d,gr_1st = None,check_var_name_present = True):

    #do_addtimedim = True

    nav_lon_var_mat = ['nav_lon'.upper(),'lon'.upper(),'longitude'.upper(),'TLON'.upper(),'nav_lon_grid_T'.upper(),'nav_lon_grid_U'.upper(),'nav_lon_grid_V'.upper()]
    nav_lat_var_mat = ['nav_lat'.upper(),'lat'.upper(),'latitude'.upper(),'TLAT'.upper(),'nav_lat_grid_T'.upper(),'nav_lat_grid_U'.upper(),'nav_lat_grid_V'.upper()]
    time_varname_mat = ['time_counter'.upper(),'time'.upper(),'t'.upper()]
        # match def resample_xarray() to time_varname_mat, until generalised. 

    nav_lon_varname,nav_lat_varname,time_varname = None, None, None
    # check name of lon and lat ncvar in data.
    # cycle through variables and if it is a possibn le varibable name, use it

    nav_lon_varname_dict = {}
    nav_lat_varname_dict = {}
    time_varname_dict = {}
    

    for tmp_datstr in ncvar_d.keys():#Dataset_lst:              
        #nav_lon_varname_dict[tmp_datstr] = {}            
        #nav_lat_varname_dict[tmp_datstr] = {}            
        time_varname_dict[tmp_datstr] = {}


        nav_lon_varname_dict[tmp_datstr] = None
        nav_lat_varname_dict[tmp_datstr] = None
 
        for tmpgrid in ncvar_d[tmp_datstr].keys():   
            #pdb.set_trace()
            #for ncvar in ncvar_d[tmp_datstr]['T']: 
            for ncvar in ncvar_d[tmp_datstr][tmpgrid]: 
                if ncvar.upper() in nav_lon_var_mat: nav_lon_varname = ncvar
                if ncvar.upper() in nav_lat_var_mat: nav_lat_varname = ncvar
                if ncvar.upper() in time_varname_mat: time_varname = ncvar

            
            #if nav_lon_varname not in ncvar_d['Dataset 1']['T']:
            #    pdb.set_trace()

            #nav_lon_varname_dict[tmp_datstr][tmpgrid] = nav_lon_varname
            #nav_lat_varname_dict[tmp_datstr][tmpgrid] = nav_lat_varname
            time_varname_dict[tmp_datstr][tmpgrid] = time_varname
            #if tmpgrid in ['T','T_1']:

            '''
            if tmpgrid == gr_1st:
                nav_lon_varname_dict[tmp_datstr] = nav_lon_varname
                nav_lat_varname_dict[tmp_datstr] = nav_lat_varname
            '''

            update_lon_lat = False
            if gr_1st is None:
                if tmpgrid in ['T','T_1']:
                    update_lon_lat = True
                    #nav_lon_varname_dict[tmp_datstr] = nav_lon_varname
                    #nav_lat_varname_dict[tmp_datstr] = nav_lat_varname
            else:
                if tmpgrid == gr_1st:
                    update_lon_lat = True
                    #nav_lon_varname_dict[tmp_datstr] = nav_lon_varname
                    #nav_lat_varname_dict[tmp_datstr] = nav_lat_varname

            # if LBC is the first config,  
            #if (update_lon_lat == False)&(nav_lon_varname_dict[tmp_datstr] is None):
            #    update_lon_lat = False


            if update_lon_lat == True:
                nav_lon_varname_dict[tmp_datstr] = nav_lon_varname
                nav_lat_varname_dict[tmp_datstr] = nav_lat_varname


            if check_var_name_present:
                if nav_lon_varname is None:
                    print('\ncreate_ncvar_lon_lat_time_dict: nav_lon_varname is None,%s,%s\n\n'%(tmp_datstr,tmpgrid))
                    pdb.set_trace()
                if nav_lat_varname is None:
                    print('\ncreate_ncvar_lon_lat_time_dict: nav_lat_varname is None,%s,%s\n\n'%(tmp_datstr,tmpgrid))
                    pdb.set_trace()
                #if  not do_addtimedim:
                if time_varname is None:
                    print('\ncreate_ncvar_lon_lat_time_dict: time_varname is None,%s,%s\n\n'%(tmp_datstr,tmpgrid))
                    pdb.set_trace()
    

    if check_var_name_present:
        if nav_lon_varname is None:
            print('\ncreate_ncvar_lon_lat_time_dict: nav_lon_varname is None\n\n')
            pdb.set_trace()
        if nav_lat_varname is None:
            print('\ncreate_ncvar_lon_lat_time_dict: nav_lat_varname is None\n\n')
            pdb.set_trace()
        #if  not do_addtimedim:
        if time_varname is None:
            print('\ncreate_ncvar_lon_lat_time_dict: time_varname is None\n\n')
            pdb.set_trace()
    
    #pdb.set_trace()

    #for tmp_datstr in ncvar_d.keys():#Dataset_lst:   
    #    if nav_lon_varname_dict[tmp_datstr] is None: pdb.set_trace()

    return nav_lon_varname_dict,nav_lat_varname_dict,time_varname_dict,nav_lon_var_mat,nav_lat_var_mat,time_varname_mat






#def create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutxind,cutyind,cutout_data):

def create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutout_d,gr_1st = 'T'):

    
    lon_d,lat_d = {},{}

    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        #pdb.set_trace()

        nav_lat_varname = nav_lat_varname_dict[tmp_datstr]
        nav_lon_varname = nav_lon_varname_dict[tmp_datstr]

        tmp_configd = configd[th_d_ind]
        if tmp_configd is None: tmp_configd = configd[1]
        # load nav_lat and nav_lon

        cutxind,cutyind,cutout_data = cutout_d[th_d_ind]['cutxind'],cutout_d[th_d_ind]['cutyind'],cutout_d[th_d_ind]['do_cutout']

        if tmp_configd.upper() in ['ORCA025','ORCA025EXT','ORCA12','ORCA025ICE','ORCA12ICE']: 

            #lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0])
            #lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0])
            lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])
            lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])
            
            # Fix Longitude, to be between -180 and 180.
            fixed_nav_lon = lon_d[th_d_ind].copy()
            for i, start in enumerate(np.argmax(np.abs(np.diff(lon_d[th_d_ind])) > 180, axis=1)):            fixed_nav_lon[i, start+1:] += 360
            fixed_nav_lon -=360
            fixed_nav_lon[fixed_nav_lon<-287.25] +=360
            fixed_nav_lon[fixed_nav_lon>73] -=360
            lon_d[th_d_ind] = fixed_nav_lon.copy()


            #lat_d[th_d_ind] = np.ma.array(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            #lon_d[th_d_ind] = np.ma.array(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            lat_d[th_d_ind] = np.ma.array(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            lon_d[th_d_ind] = np.ma.array(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])


        else:
        #elif tmp_configd.upper() in ['CO9P2','AMM15','AMM7','GULF18','CO9P2_LBC','AMM15_LBC','AMM7_LBC','GULF18_LBC']:
            #elif tmp_configd.upper() in ['CO9P2']: 
            # when loading a year of AMM15 3d Daily Mean files, 7/17mins initialisatoin time is to load lat lons!
            # as was taking them from the data... instead, add amm15 and amm7 to CO9p2 and load from mesh file.

            '''

            before:
            Initialisation time 07 - 08: 0:07:51.438504 - created ncvar lon lat time - created lon lat dict 
            ...

            after:
            Initialisation: total: 0:17:43.025076
            ======================================
            Initialisation time 07 - 08: 0:00:00.269349 - created ncvar lon lat time - created lon lat dict 
            ...
            Initialisation: total: 0:10:48.254651

            
            '''

            #lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0])
            #lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0])
            lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])
            lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])

            if tmp_configd.upper() in ['CO9P2','AMM15']: 
                lat_d['amm15'] = np.ma.array(lat_d[th_d_ind].copy())
                lon_d['amm15'] = np.ma.array(lon_d[th_d_ind].copy())
            

            lat_d[th_d_ind] = np.ma.array(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            lon_d[th_d_ind] = np.ma.array(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])

            #pdb.set_trace()

        """
        else:
            #pdb.set_trace()
            # find the dimension of the latitude variable.
            #nav_lat_dims = list(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].dims)
            nav_lat_dims = list(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname].dims)

            # if the latitude variable has a time dimension, remove it
            nav_lat_dims_no_time = nav_lat_dims.copy()
            inc_time = False
            #if ncdim_d[tmp_datstr]['T']['t'] in nav_lat_dims_no_time:
            if ncdim_d[tmp_datstr][gr_1st]['t'] in nav_lat_dims_no_time:
                #nav_lat_dims_no_time.remove(ncdim_d[tmp_datstr]['T']['t'] )
                nav_lat_dims_no_time.remove(ncdim_d[tmp_datstr][gr_1st]['t'] )
                inc_time = True

            #if len(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].shape) == 2:

            #pdb.set_trace()
                
            # if lat/lon only have 1 dimension (e.g. NWSPPE), use meshgrid to turn them into matrices.
            if len(nav_lat_dims_no_time) == 1:
                #pdb.set_trace()
                #lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname].load())
                #lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].load())
                #tmp1dlon = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname].load())
                #tmp1dlat = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].load())
                tmp1dlon = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lon_varname].load())
                tmp1dlat = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname].load())
                lon_d[th_d_ind],lat_d[th_d_ind] = np.meshgrid(tmp1dlon,tmp1dlat)
                
                  

            elif len(nav_lat_dims_no_time) == 2:
                if inc_time:
                    #lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][0,:,:].load())
                    #lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][0,:,:].load())
                    lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lon_varname][0,:,:].load())
                    lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname][0,:,:].load())
                    #lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())
                    #lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())

                else:
                    #lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname].load())
                    #lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].load())
                    lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lon_varname].load())
                    lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname].load())
                    #lon_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())
                    #lat_d[th_d_ind] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())


            else:
                # if only 1d lon and lat
                if inc_time:
                    #tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname].load())
                    #tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname].load())
                    tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lon_varname].load())
                    tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname].load())
                    #tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())
                    #tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())
                else:
                    #tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][0,:].load())
                    #tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][0,:].load())
                    tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lon_varname][0,:].load())
                    tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][0].variables[nav_lat_varname][0,:].load())
                    #tmp_nav_lon = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lon_varname][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())
                    #tmp_nav_lat = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][0].variables[nav_lat_varname][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]].load())

                lon_d[th_d_ind], lat_d[th_d_ind] = np.meshgrid(tmp_nav_lon,tmp_nav_lat)

            if tmp_configd.upper() in ['AMM15','CO9P2','AMM15_LBC','CO9P2_LBC']: 
                # AMM15 lon and lats are always 2d
                lat_d['amm15'] = lat_d[th_d_ind]
                lon_d['amm15'] = lon_d[th_d_ind]

            lon_d[th_d_ind] = np.ma.masked_invalid(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            lat_d[th_d_ind] = np.ma.masked_invalid(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])

        """

        #Check if any nav_lat or nav_lon have masked values (i.e. using land suppression)
        if ( ((lat_d[th_d_ind] == 0) & (lon_d[th_d_ind] == 0)).sum()>10) |  (lat_d[th_d_ind] == lon_d[th_d_ind]).sum()> 100:
            print('Several points (>10) for 0degN 0degW - suggesting land suppression - use glamt and gphit from mesh')

            #lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0])
            #lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0])
            lon_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncglamt][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])
            lat_d[th_d_ind] = np.ma.masked_invalid(rootgrp_gdept_dict[tmp_datstr].variables[ncgphit][0,cutyind[0]:cutyind[1],cutxind[0]:cutxind[1]])

            lat_d[th_d_ind] = np.ma.array(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
            lon_d[th_d_ind] = np.ma.array(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])


        #pdb.set_trace()
        if (((lat_d[th_d_ind] == 0) & (lon_d[th_d_ind] == 0)).sum()>10) |  (lat_d[th_d_ind] == lon_d[th_d_ind]).sum()> 100:
            # If there are still (0,0) pairs in nav_lat and nav_lon, coming from glamt and gphit, we can approixmate the field analytically
            print('Several points (>10) for 0degN 0degW - suggesting land suppression - calc grid mesh')
            if tmp_configd.upper() in ['AMM7']:
                # as AMM7 is a regular lat and lon grid, with a linear grid, re can use a simple linear equation, and then use mesh grid

                lon_amm7 = np.arange(-19.888889,12.99967+1/9.,1/9.)
                lat_amm7 = np.arange(40.066669,65+1/15.,1/15.)

                lon_d[th_d_ind], lat_d[th_d_ind] = np.meshgrid(lon_amm7,lat_amm7)

                lon_d[th_d_ind] = np.ma.masked_invalid(lon_d[th_d_ind])
                lat_d[th_d_ind] = np.ma.masked_invalid(lat_d[th_d_ind])

                lat_d[th_d_ind] = np.ma.array(lat_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
                lon_d[th_d_ind] = np.ma.array(lon_d[th_d_ind][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])

            if tmp_configd.upper() in ['AMM15', 'CO9P2']:
                # AMM15 is more complicated, as its on a rotated grid, however once unrotated, it can be treated as AMM7
                
                '''
                
                nav_lon = np.ma.masked_invalid(rootgrp_gdept_dict['Dataset 1'].variables[ncglamt][0])
                nav_lat = np.ma.masked_invalid(rootgrp_gdept_dict['Dataset 1'].variables[ncgphit][0])
                # unrotate the lats and lons, 
                lon_mat_unrot, lat_mat_unrot = reduce_rotamm15_grid(nav_lon,nav_lat)

                # find the linear parameters, 
                dlat_mat_unrot = np.diff(lat_mat_unrot).mean()
                dlon_mat_unrot = np.diff(lon_mat_unrot).mean()

                nlat_mat_unrot, nlon_mat_unrot = lat_mat_unrot.size, lon_mat_unrot.size
                lat_mat_unrot_0, lon_mat_unrot_0 = lat_mat_unrot[0], lon_mat_unrot[0]
                # print them
                print(lat_mat_unrot_0,nlat_mat_unrot,dlat_mat_unrot)
                print(lon_mat_unrot_0,nlon_mat_unrot,dlon_mat_unrot)

                
                lat_mat_unrot_0,nlat_mat_unrot,dlat_mat_unrot = -7.29419849537037, 1345, 0.01349999957739065
                lon_mat_unrot_0,nlon_mat_unrot,dlon_mat_unrot = -10.889596160548328, 1458, 0.013500078257954802
                lat_mat_unrot_0,nlat_mat_unrot,dlat_mat_unrot = -7.2942, 1345, 0.0135 #-7.29419849537037, 1345, 0.01349999957739065
                lon_mat_unrot_0,nlon_mat_unrot,dlon_mat_unrot = -10.889595, 1458, 0.01350#-10.889596160548328, 1458, 0.013500078257954802

                '''
                # set the linear parameters
                lat_mat_unrot_0,nlat_mat_unrot,dlat_mat_unrot = -7.29419849537037, 1345, 0.01349999957739065
                lon_mat_unrot_0,nlon_mat_unrot,dlon_mat_unrot = -10.889596160548328, 1458, 0.013500078257954802
                
                #apply a linear equation
                lat_mat_unrot_arr = lat_mat_unrot_0 + np.arange(nlat_mat_unrot)*dlat_mat_unrot
                lon_mat_unrot_arr = lon_mat_unrot_0 + np.arange(nlon_mat_unrot)*dlon_mat_unrot

                #create a grid mesh
                lon_mat_unrot_mat,lat_mat_unrot_mat = np.meshgrid(lon_mat_unrot_arr,lat_mat_unrot_arr)
                
                #then rotate it
                lon_mat_unrot_mat_rot,lat_mat_unrot_mat_rot = rotated_grid_to_amm15(lon_mat_unrot_mat,lat_mat_unrot_mat)
                lat_d['amm15'] = lat_d[th_d_ind].copy()
                lon_d['amm15'] = lon_d[th_d_ind].copy()

                #thin the lats and lons. 
                lat_d[th_d_ind] = np.ma.array(lat_mat_unrot_mat_rot[thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])
                lon_d[th_d_ind] = np.ma.array(lon_mat_unrot_mat_rot[thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']])


    return lon_d,lat_d









def resample_xarray(xarr_dict,resample_freq,time_varname_dict):
    #e.g. resample_freq = '1m', '5d', requires "DatetimeIndex, TimedeltaIndex or PeriodIndex," , doesn't work with dummy dates (i.e. increments)
    #xarr_dict['Dataset 1']['T'][0] = xarr_dict['Dataset 1']['T'][0].resample(time_counter = '1m').mean()

    #check if resample_freq string:
    # remove digits from resample_freq, so left with the letters - check if they are M, Y, D, Q etc.)
    non_digit_resample_freq = ''.join([ss for ss in resample_freq if not  ss.isdigit() ])

    if non_digit_resample_freq not in ['M', 'Y', 'D', 'Q']:
        print('\n\n%s may not be a valid resampling string. Typically a number with a letter for the number of days, months, quarters or years - D, M, Q, Y'%resample_freq)
        print('Will try proceeding\n\n\n')


    for tmp_datstr in xarr_dict.keys():
        #time_varname = time_varname_dict[tmp_datstr]
        for tmpgrid in xarr_dict[tmp_datstr].keys():
            time_varname = time_varname_dict[tmp_datstr][tmpgrid]
            for xarlii in range(len(xarr_dict[tmp_datstr][tmpgrid])):
                if time_varname == 'time_counter':
                    xarr_dict[tmp_datstr][tmpgrid][xarlii] = xarr_dict[tmp_datstr][tmpgrid][xarlii].resample(time_counter = resample_freq).mean()
                elif time_varname == 'time':
                    #pdb.set_trace()
                    xarr_dict[tmp_datstr][tmpgrid][xarlii] = xarr_dict[tmp_datstr][tmpgrid][xarlii].resample(time = resample_freq).mean()
                else:
                    print('Resample only coded for time_counter and time, needs to be generalised.  Your time var is %s'%time_varname)
                    pdb.set_trace()


    return xarr_dict


def extract_time_from_xarr(xarr_dict_in,ex_fname_in,time_varname_in,t_dim,date_in_ind,date_fmt,ti,verbose_debugging):

    '''
    
    time_datetime,time_datetime_since_1970,ntime = extract_time_from_xarr(xarr_dict['Dataset 1']['T'],fname_dict['Dataset 1']['T'][0],date_in_ind,date_fmt,verbose_debugging)
    '''
    #pdb.set_trace()
    
    #print ('xarray start reading nctime',datetime.now())



    #if both time and time_counter used, (as in increments), use time_counter
    if ('time' in xarr_dict_in[0].variables.keys()) & ('time_counter' in xarr_dict_in[0].variables.keys()):
        time_varname = 'time_counter'
    else:
        time_varname = time_varname_in
    #pdb.set_trace()
    # Extract time variable (with attributes) from xarray


    

    try:
        nctime = xarr_dict_in[0].variables[time_varname]
    except:


        pdb.set_trace()


    try:

        #xarray nctime to datetime:
        #xarray nctime to timestamp
        if isinstance(nctime.to_numpy()[0],np.datetime64):
            nctime_calendar_type = 'greg'
            nctime_timestamp = ( nctime.to_numpy() - np.datetime64('1970-01-01T00:00:00'))/ np.timedelta64(1,'s')

            time_datetime = np.array([datetime.utcfromtimestamp(ss) for ss in nctime_timestamp])
            
            #time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])
            time_datetime_since_1970 = nctime_timestamp/86400
        elif isinstance(nctime.to_numpy()[0],cftime.Datetime360Day):
            nctime_calendar_type = '360_day'

            time_datetime_since_1970 = np.array([ss.year + (ss.month-1)/12 + (ss.day-1)/360 for ss in np.array(nctime)])
            time_datetime = time_datetime_since_1970
        else: 
            nctime_calendar_type = 'greg'
            print('Interpreted time info class:',type(nctime.to_numpy()[0]))
            # create dummy time data, starting with today, and going forward one day for each time.

            # today
            tmpdatetime_now = datetime.now()
            tmpdate_now = datetime(tmpdatetime_now.year, tmpdatetime_now.month, tmpdatetime_now.day)

            #time values in array: if all are zero, increment, otherwise use. 
            nctime_in_array = nctime.to_numpy()
            if (nctime_in_array == 0).all():
                nctime_in_array = np.arange( len(nctime))
            
            time_datetime = np.array([tmpdate_now + timedelta(days = int(i_i)) for i_i in nctime_in_array])
            time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])



        ntime = time_datetime.size
        #if ('calendar' in nctime.attrs.keys()):
        #    nctime_calendar_type = nctime.attrs['calendar']
        #else:
        #    nctime_calendar_type = 'greg'


        if date_in_ind is not None:
            date_in_ind_datetime = datetime.strptime(date_in_ind,date_fmt)
            date_in_ind_datetime_timedelta = np.array([(ss - date_in_ind_datetime).total_seconds() for ss in time_datetime])
            ti = np.abs(date_in_ind_datetime_timedelta).argmin()
            if verbose_debugging: print('Setting ti from date_in_ind (%s): ti = %i (%s). '%(date_in_ind,ti, time_datetime[ti]), datetime.now())

        return time_datetime,time_datetime_since_1970,ntime,ti, nctime_calendar_type
    




    except:
        print('\n\n\n\nTrying new time processing failed\n\n\n\n\n')
        #pdb.set_trace()

    # if all times are 0, and no time_origin, suggests time is not set for these input files, so make dummy time data

    # if ((nctime.load()[:] == 0).all()) & ('time_origin' not in nctime.attrs.keys()):

    #if time_origin not in time attributesxarr_dict_in

    use_time_units_for_origin = False
    if ('time_origin' not in nctime.attrs.keys()):
        pdb.set_trace()
        if ('units' in nctime.attrs.keys()):
            use_time_units_for_origin = True
        else:
            #if all time values are 0.
            all_time_0 = (nctime.load()[:] == 0).all()

            if all_time_0:
                print('No time origin and all time values == 0')
            else:
                print('No time origin but some time values != 0')
                print('Setting all_time_0 = True')
                all_time_0 = True

            if all_time_0:

                # add time data as daily from the current day.
                time_datetime = np.array([datetime(datetime.now().year, datetime.now().month, datetime.now().day) + timedelta(days = i_i) for i_i in range( xarr_dict_in[0].dims[t_dim])])
                print("xarr_dict_in[0].dims[t_dim]")
                #except:
                #    time_datetime = np.array([datetime(datetime.now().year, datetime.now().month, datetime.now().day) + timedelta(days = i_i) for i_i in range( xarr_dict_in[0][0].dims[t_dim])])
                #    print("xarr_dict_in[0][0].dims[t_dim]")
                time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])

                if date_in_ind is not None: ti = 0
                ntime = time_datetime.size
                nctime_calendar_type = 'greg'


                return time_datetime,time_datetime_since_1970,ntime,ti, nctime_calendar_type
            



    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
        
    # xarray appears to be inconsistent in how you access calendars, so rather risking crashing with some files,
    #   or using a try with 'calendar' in nctime.to_index()._attributes and nctime_calendar_type = nctime.to_index().calendar
    #   we simply check the type of the xarray datetime array
        
    nctime_calendar_type = None
    
    if str(type(nctime.load().data[0])).find('Datetime360Day')>0:
        nctime_calendar_type = '360_day'
    else:
        nctime_calendar_type = 'greg'

    

    # If there is a time_origin use it, otherwise make a dummy time_origin
    if 'time_origin' in nctime.attrs.keys():
        nc_time_origin = nctime.attrs['time_origin']
    else:
        if use_time_units_for_origin:
            nc_units = nctime.attrs['units']
            pdb.set_trace()
            #nc_time_origin = 
        else:
            nc_time_origin = '1980-01-01 00:00:00'
            print('No time origin set - set to 1/1/1980. Other Time parameters likely to be missing')
   


    #calculate time_datetime_since_1970 differently for 360/360_day and greg
    if  nctime_calendar_type in ['360','360_day']:
        # if 360 days

        time_datetime_since_1970 = np.array([ss.year + (ss.month-1)/12 + (ss.day-1)/360 for ss in np.array(nctime)])
        time_datetime = time_datetime_since_1970
    else:
        # if gregorian        
        sec_since_origin = [float(ii.data - np.datetime64(nc_time_origin))/1e9 for ii in nctime]
        time_datetime_cft = num2date(sec_since_origin,units = 'seconds since ' + nc_time_origin,calendar = 'gregorian') #nctime.calendar)

        time_datetime = np.array([datetime(ss.year, ss.month,ss.day,ss.hour,ss.minute) for ss in time_datetime_cft])
        time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])


    # if the input time index is given as a date, convert it to the time index. 
    if date_in_ind is not None:
        date_in_ind_datetime = datetime.strptime(date_in_ind,date_fmt)
        date_in_ind_datetime_timedelta = np.array([(ss - date_in_ind_datetime).total_seconds() for ss in time_datetime])
        ti = np.abs(date_in_ind_datetime_timedelta).argmin()
        if verbose_debugging: print('Setting ti from date_in_ind (%s): ti = %i (%s). '%(date_in_ind,ti, time_datetime[ti]), datetime.now())


    ntime = time_datetime.size
    #print(nctime_calendar_type,nc_time_origin)

    return time_datetime,time_datetime_since_1970,ntime,ti, nctime_calendar_type





def load_ops_2D_xarray(OPSfname,vartype, nlon = 1458, nlat = 1345,  excl_qc = False, timing_in = 1):


    timing = False
    timing_details = False

    if timing_in >0:
        timing = True
    if timing_in>1:
        timing_details =True
    tstart = datetime.now()

    if timing: 
        tim_lst = []

    if vartype == 'ChlA':
        stat_type_lst = [389]
        ops_qc_var_good_1_lst = ['OBSERVATION_QC','SLCHLTOT_QC']#,'SLA_LEVEL_QC']
        iobsi = 'SLCHLTOT_IOBSI'
        iobsj = 'SLCHLTOT_IOBSJ'
        ops_output_var_mat = ['LONGITUDE', 'LATITUDE', 'DEPTH', 'JULD', 'SLCHLTOT_OBS', 'SLCHLTOT_Hx', 'SLCHLTOT_STD']#,'SLCHLTOT_GRID','MDT']
        ops_output_ind_mat = ['SLCHLTOT_IOBSI', 'SLCHLTOT_IOBSJ', 'SLCHLTOT_IOBSK']
    elif vartype == 'SST_ins':
        stat_type_lst = [50,53,55]
        ops_qc_var_good_1_lst = ['OBSERVATION_QC','SST_QC']#,'SST_LEVEL_QC']
        iobsi = 'SST_IOBSI'
        iobsj = 'SST_IOBSJ'
        ops_output_var_mat = ['LONGITUDE', 'LATITUDE', 'DEPTH', 'JULD', 'SST_OBS', 'SST_Hx', 'SST_STD']
        ops_output_ind_mat = ['SST_IOBSI', 'SST_IOBSJ', 'SST_IOBSK']
    elif vartype == 'SST_sat':
        stat_type_lst = np.arange(50)
        ops_qc_var_good_1_lst = ['OBSERVATION_QC','SST_QC']#,'SST_LEVEL_QC']
        iobsi = 'SST_IOBSI'
        iobsj = 'SST_IOBSJ'
        ops_output_var_mat = ['LONGITUDE', 'LATITUDE', 'DEPTH', 'JULD', 'SST_OBS', 'SST_Hx', 'SST_STD']
        ops_output_ind_mat = ['SST_IOBSI', 'SST_IOBSJ', 'SST_IOBSK']

    elif vartype == 'SLA':
        stat_type_lst = [  61,   65,  262,  441, 1005]
        ops_qc_var_good_1_lst = ['OBSERVATION_QC','SLA_QC']#,'SLA_LEVEL_QC']
        iobsi = 'SLA_IOBSI'
        iobsj = 'SLA_IOBSJ'
        ops_output_var_mat = ['LONGITUDE', 'LATITUDE', 'DEPTH', 'JULD', 'SLA_OBS', 'SLA_Hx', 'SLA_SSH','MDT']#,'SLA_GRID','MDT']
        ops_output_ind_mat = ['SLA_IOBSI', 'SLA_IOBSJ', 'SLA_IOBSK']

    if timing: tim_lst.append(('Selected options',datetime.now()))

    #https://code.metoffice.gov.uk/trac/ops/browser/main/trunk/src/code/OpsMod_SatSST/Ops_SSTFeedbackWriteNetCDF.inc#L204


    #50 = ship, 53 = drifting buoy, 55 = moored buoy.

    # OPSfname = '/scratch/hadjt/SSF/flx/fdbk.obsop.daym2/mi-bb024_amm15finalps45trial4_ctrl/20210408_sst_UnBiasCorrfb_01.nc'
 

    root_x = xarray.open_dataset(OPSfname, engine="netcdf4", decode_cf= False)


    if timing: tim_lst.append(('Opened file with xarray',datetime.now()))
    ops_var_mat =[ss for ss in root_x.variables.keys() ]
    if timing: tim_lst.append(('Read Var names',datetime.now()))
    ops_dim_mat = [ss for ss in root_x.dims.keys() ]
    if timing: tim_lst.append(('Read Dim names',datetime.now()))



    ops_dim_dict = {}
    for ss in ops_dim_mat: ops_dim_dict[ss] = root_x.dims[ss]

    if timing: tim_lst.append(('Read Dim sizes',datetime.now()))




    # find obs where all qc flags are zero.

    #QC flags, where 0 indicates good data
    ops_qc_var_good_0_lst = ['DEPTH_QC','POSITION_QC','JULD_QC']

    #QC flags, where 1 indicates good data

    #ops_qc_var_good_1_lst = ['OBSERVATION_QC','SLCHLTOT_QC']#,'SLA_LEVEL_QC']

    #pdb.set_trace()

    comb_qc_flag = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')

    for ss in ops_qc_var_good_0_lst:  comb_qc_flag += (root_x.variables[ss].load().data[:].ravel() != 0).astype('int')
    if timing: tim_lst.append(('Read default QC flags',datetime.now()))
    #print(100*comb_qc_flag.mean())
    for ss in ops_qc_var_good_1_lst:  comb_qc_flag += (root_x.variables[ss].load().data[:].ravel() != 1).astype('int')
    if timing: tim_lst.append(('Read specific QC flags',datetime.now()))
    #print(100*comb_qc_flag.mean())
    #comb_qc_flag += (rootgrp.variables['OBSERVATION_QC'][:].ravel() != 1).astype('int')



    #pdb.set_trace()
    # Need to read about nemovar 'NEMOVAR flag conventions'
    #comb_qc_flag = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')
    
    
    #pdb.set_trace()
    # find obs with correct station types.
    # obs  station types.
    #stat_type = np.array(chartostring(rootgrp.variables['STATION_TYPE'][:])).astype('float')
    stat_type = np.array(chartostring(root_x.variables['STATION_TYPE'].load().data[:])).astype('float')
    if timing: tim_lst.append(('Read Station types',datetime.now()))

    #pdb.set_trace()
    stat_type_ind = np.isin(stat_type,stat_type_lst)
    if timing: tim_lst.append(('Selected Station types',datetime.now()))

    #pdb.set_trace()

    # find obs within domain

    #loc_ind = (root_x.variables['SLCHLTOT_IOBSI'].load().data[:]>=0) & (root_x.variables['SLCHLTOT_IOBSI'].load().data[:]<nlon) & (root_x.variables['SLCHLTOT_IOBSJ'].load().data[:]>=0) & (root_x.variables['SLCHLTOT_IOBSJ'].load().data[:]<nlat)
    loc_ind = (root_x.variables[iobsi].load().data[:]>=0) & (root_x.variables[iobsi].load().data[:]<nlon) & (root_x.variables[iobsj].load().data[:]>=0) & (root_x.variables[iobsj].load().data[:]<nlat)

    if timing: tim_lst.append(('Selected location indices types',datetime.now()))
    
    # combine all indices

    if excl_qc:
        comb_ind =  stat_type_ind & loc_ind
    else:
        comb_ind = (comb_qc_flag==0) & stat_type_ind & loc_ind

        

    if timing: tim_lst.append(('Combined QC flags',datetime.now()))

    ops_output_dict = {}
    for ss in ops_output_var_mat: 
        ops_output_dict[ss] = np.ma.masked_equal(root_x.variables[ss][comb_ind].load().data[:],root_x.variables[ss].attrs['_Fillvalue'])
        if timing: tim_lst.append(('Read var:'+ss,datetime.now()))
    for ss in ops_output_ind_mat:
        ops_output_dict[ss] = root_x.variables[ss][comb_ind].load().data[:]
        if timing: tim_lst.append(('Read ind:'+ss,datetime.now()))
  
    if timing: tim_lst.append(('Read in Ind and vars',datetime.now()))


    JULD_REFERENCE = datetime.strptime(str(chartostring(root_x.variables['JULD_REFERENCE'].load().data[:])),'%Y%m%d%H%M%S')


    if timing: tim_lst.append(('Converted Juld Ref to date time',datetime.now()))



    ops_output_dict['JULD_datetime'] = np.array([JULD_REFERENCE + timedelta(ss) for ss in root_x.variables['JULD'][comb_ind].load().data[:]])


    if timing: tim_lst.append(('JULD_datetime',datetime.now()))

    ops_output_dict['STATION_IDENTIFIER'] = np.array([str(ss) for ss in chartostring(root_x.variables['STATION_IDENTIFIER'][comb_ind,:].load().data[:])])

    if timing: tim_lst.append(('STATION_IDENTIFIER',datetime.now()))
    ops_output_dict['STATION_TYPE'] = stat_type[comb_ind]
    if timing: tim_lst.append(('STATION_TYPE',datetime.now()))


    root_x.close()

    if timing: tim_lst.append(('Finished',datetime.now()))

    if timing:
        if timing_details:
            tprev = tstart
            for (tmplab,tmptime) in tim_lst: 
                print('    %35s'%tmplab, tmptime, tmptime - tstart, tmptime-tprev)
                tprev = tmptime

            print()
        print('    Total time:',tim_lst[-1][1]-tstart)







    if  vartype =='SST_ins' :
        ops_output_dict['OBS'] = ops_output_dict['SST_OBS'] 
        ops_output_dict['MOD_HX'] = ops_output_dict['SST_Hx'] 
    elif  vartype =='SST_sat' :
        ops_output_dict['OBS'] = ops_output_dict['SST_OBS'] 
        ops_output_dict['MOD_HX'] = ops_output_dict['SST_Hx'] 
    elif  vartype =='SLA' :
        ops_output_dict['OBS'] = ops_output_dict['SLA_OBS']  + ops_output_dict['MDT']
        ops_output_dict['MOD_HX'] = ops_output_dict['SLA_SSH']  
    elif  vartype == 'ChlA':
        ops_output_dict['OBS'] = 10**ops_output_dict['SLCHLTOT_OBS']
        ops_output_dict['MOD_HX'] = 10**ops_output_dict['SLCHLTOT_Hx'] 










    return ops_output_dict



def load_ops_prof_TS(OPSfname, TS_str_in,stat_type_lst = None,nlon = 1458, nlat = 1345, excl_qc = False):
    '''
    stat_type_lst = [50,53,55]
    nlon = 1458
    nlat = 1345
    excl_qc = True    
    '''

    if TS_str_in.upper() in ['T','POTM','VOTEMPER','TEMPERATURE']:
        TnotS = True
    elif TS_str_in.upper() in ['S','PSAL','VOSALINE','SALINITY']:
        TnotS = False
    else:
        print('TS_str_in must be T or S, not ',TS_str_in)
        pdb.set_trace()


    #https://code.metoffice.gov.uk/trac/ops/browser/main/trunk/src/code/OpsMod_SatSST/Ops_SSTFeedbackWriteNetCDF.inc#L204


    #50 = ship, 53 = drifting buoy, 55 = moored buoy.

    # OPSfname = '/scratch/hadjt/SSF/flx/fdbk.obsop.daym2/mi-bb024_amm15finalps45trial4_ctrl/20210408_sst_UnBiasCorrfb_01.nc'
 

    
    rootgrp = Dataset(OPSfname, 'r', format='NETCDF4')
    ops_var_mat = [ ss for ss in rootgrp.variables.keys() ]
    ops_dim_mat = [ ss for ss in rootgrp.dimensions.keys() ]


    #ops_var_dict = {}
    #for ss in ops_var_mat: ops_var_dict[ss] = rootgrp.variables[ss][:]
    ops_dim_dict = {}
    for ss in ops_dim_mat: ops_dim_dict[ss] = rootgrp.dimensions[ss].size


    '''
    'VARIABLES'
    'ENTRIES'
    'EXTRA'
    'STATION_IDENTIFIER'
    'STATION_TYPE'
    'LONGITUDE'
    'LATITUDE'
    'DEPTH'
    'DEPTH_QC'
    'DEPTH_QC_FLAGS'
    'JULD'
    'JULD_REFERENCE'
    'OBSERVATION_QC'
    'OBSERVATION_QC_FLAGS'
    'POSITION_QC'
    'POSITION_QC_FLAGS'
    'JULD_QC'
    'JULD_QC_FLAGS'
    'ORIGINAL_FILE_INDEX'
    'POTM_OBS'
    'POTM_Hx'
    'POTM_QC'
    'POTM_QC_FLAGS'
    'POTM_LEVEL_QC'
    'POTM_LEVEL_QC_FLAGS'
    'POTM_IOBSI'
    'POTM_IOBSJ'
    'POTM_IOBSK'
    'POTM_GRID'
    'PSAL_OBS'
    'PSAL_Hx'
    'PSAL_QC'
    'PSAL_QC_FLAGS'
    'PSAL_LEVEL_QC'
    'PSAL_LEVEL_QC_FLAGS'
    'PSAL_IOBSI'
    'PSAL_IOBSJ'
    'PSAL_IOBSK'
    'PSAL_GRID'
    'TEMP'
    '''
    '''
    # find obs where all qc flags are zero.

    #QC flags, where 0 indicates good data
    ops_qc_var_good_0_lst = ['DEPTH_QC','POSITION_QC','JULD_QC']

    #QC flags, where 1 indicates good data
    #ops_qc_var_good_1_lst = ['OBSERVATION_QC','SST_QC']#,'SST_LEVEL_QC']
    #ops_qc_var_good_1_lst = ['OBSERVATION_QC','POTM_QC','PSAL_QC']#,'SST_LEVEL_QC']


    ops_qc_var_good_POTM_lst = ['OBSERVATION_QC','POTM_QC']#,'SST_LEVEL_QC']
    ops_qc_var_good_PSAL_lst = ['OBSERVATION_QC','PSAL_QC']#,'SST_LEVEL_QC']


    #pdb.set_trace()
    comb_qc_flag = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')
    for ss in ops_qc_var_good_0_lst:  comb_qc_flag =  comb_qc_flag + (rootgrp.variables[ss][:] != 0).astype('int').T
    #for ss in ops_qc_var_good_1_lst:  comb_qc_flag = comb_qc_flag +  (rootgrp.variables[ss][:] != 1).astype('int').T


    comb_qc_flag_POTM = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')
    comb_qc_flag_PSAL = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')

    for ss in ops_qc_var_good_0_lst:  comb_qc_flag_POTM =  comb_qc_flag_POTM + (rootgrp.variables[ss][:] != 0).astype('int').T
    for ss in ops_qc_var_good_POTM_lst:  comb_qc_flag_POTM = comb_qc_flag_POTM +  (rootgrp.variables[ss][:] != 1).astype('int').T
    for ss in ops_qc_var_good_0_lst:  comb_qc_flag_PSAL =  comb_qc_flag_PSAL + (rootgrp.variables[ss][:] != 0).astype('int').T
    for ss in ops_qc_var_good_PSAL_lst:  comb_qc_flag_PSAL = comb_qc_flag_PSAL +  (rootgrp.variables[ss][:] != 1).astype('int').T

    '''


    comb_qc_flag = np.zeros((ops_dim_dict['N_OBS'],ops_dim_dict['N_LEVELS']), dtype = 'int')
    comb_qc_flag =  comb_qc_flag + (rootgrp.variables['DEPTH_QC'][:] != 1).astype('int')  # good data (1) added as a zero
    comb_qc_flag =  comb_qc_flag + np.tile((rootgrp.variables['POSITION_QC'][:] != 1).astype('int'),(ops_dim_dict['N_LEVELS'],1)).T # good data (1) added as a zero
    comb_qc_flag =  comb_qc_flag + np.tile((rootgrp.variables['OBSERVATION_QC'][:] != 0).astype('int'),(ops_dim_dict['N_LEVELS'],1)).T # good data (0) added as a zero


    #comb_qc_flag_POTM = comb_qc_flag.copy()
    #comb_qc_flag_PSAL = comb_qc_flag.copy()

    comb_qc_flag_POTM = comb_qc_flag.copy() + (rootgrp.variables['POTM_LEVEL_QC'][:] != 1).astype('int') # good data (1) added as a zero
    comb_qc_flag_PSAL = comb_qc_flag.copy() + (rootgrp.variables['PSAL_LEVEL_QC'][:] != 1).astype('int') # good data (1) added as a zero
    comb_qc_flag_POTM =  comb_qc_flag_POTM.copy() + np.tile((rootgrp.variables['POTM_QC'][:] != 1).astype('int'),(ops_dim_dict['N_LEVELS'],1)).T  # good data (1) added as a zero
    comb_qc_flag_PSAL =  comb_qc_flag_PSAL.copy() + np.tile((rootgrp.variables['PSAL_QC'][:] != 1).astype('int'),(ops_dim_dict['N_LEVELS'],1)).T  # good data (1) added as a zero


    #comb_qc_flag_POTM_2d = (comb_qc_flag_POTM==0).any(axis = 1)
    #comb_qc_flag_PSAL_2d = (comb_qc_flag_PSAL==0).any(axis = 1)

    #pdb.set_trace()
    # Need to read about nemovar 'NEMOVAR flag conventions'
    #comb_qc_flag = np.zeros((ops_dim_dict['N_OBS']), dtype = 'int')
    
    
    #pdb.set_trace()
    # find obs with correct station types.
    # obs  station types.
    stat_type = np.array(chartostring(rootgrp.variables['STATION_TYPE'][:])).astype('float')
    #stat_type_ind = np.isin(stat_type,stat_type_lst)

    #pdb.set_trace()

    # find obs within domain

    if TnotS:
        loc_ind = (rootgrp.variables['POTM_IOBSI'][:]>=0) & (rootgrp.variables['POTM_IOBSI'][:]<nlon) & (rootgrp.variables['POTM_IOBSJ'][:]>=0) & (rootgrp.variables['POTM_IOBSJ'][:]<nlat)
    else:
        loc_ind = (rootgrp.variables['PSAL_IOBSI'][:]>=0) & (rootgrp.variables['PSAL_IOBSI'][:]<nlon) & (rootgrp.variables['PSAL_IOBSJ'][:]>=0) & (rootgrp.variables['PSAL_IOBSJ'][:]<nlat)
    
  
    
    if excl_qc:
        comb_ind = loc_ind
        comb_ind_T = loc_ind
        comb_ind_S = loc_ind
    else:
        comb_ind = loc_ind
        comb_ind_T = (comb_qc_flag_POTM==0)  & loc_ind
        comb_ind_S = (comb_qc_flag_PSAL==0)  & loc_ind

        
    if TnotS:
        comb_ind = comb_ind_T
    else:
        comb_ind = comb_ind_S


    #ops_output_var_3d_mat = ['DEPTH', 'OBSERVATION_QC','POTM_QC','PSAL_QC','POTM_OBS', 'POTM_Hx', 'PSAL_OBS', 'PSAL_Hx',]
    ops_output_var_3d_mat = ['DEPTH', 'OBSERVATION_QC']
    ops_output_var_3d_T_mat =['POTM_QC','POTM_OBS', 'POTM_Hx']
    ops_output_var_3d_S_mat =['PSAL_QC','PSAL_OBS', 'PSAL_Hx',]
    #ops_output_ind_3d_mat = ['POTM_IOBSK','PSAL_IOBSK']
    ops_output_ind_3d_T_mat = ['POTM_IOBSK']
    ops_output_ind_3d_S_mat = ['PSAL_IOBSK']
    ops_output_var_2d_mat = ['LONGITUDE', 'LATITUDE',  'JULD','POSITION_QC','JULD_QC']
    #ops_output_ind_2d_mat = ['POTM_IOBSI', 'POTM_IOBSJ','PSAL_IOBSI', 'PSAL_IOBSJ']
    ops_output_ind_2d_mat = ['DEPTH_QC']
    ops_output_ind_2d_T_mat = ['POTM_IOBSI', 'POTM_IOBSJ']
    ops_output_ind_2d_S_mat = ['PSAL_IOBSI', 'PSAL_IOBSJ']
    ops_output_dict = {}
   # for ss in ops_output_var_mat: ss, rootgrp.variables[ss][comb_ind].shape
    
    #pdb.set_trace()
    '''
    
    ('DEPTH', (8, 556))
('OBSERVATION_QC', (8,))
('LONGITUDE', (9,))
('LATITUDE', (9,))
('JULD', (9,))
('POSITION_QC', (9,))
('JULD_QC', (9,))
('DEPTH_QC', (9, 556))
('POTM_QC', (8,))
('POTM_OBS', (8, 556))
('POTM_Hx', (8, 556))
('POTM_IOBSK', (8, 556))
('POTM_IOBSI', (9,))
('POTM_IOBSJ', (9,))
('JULD_datetime', (8,))
('STATION_TYPE', (9,))
('STATION_IDENTIFIER', (8,))

    
    '''




    #pdb.set_trace()
    for ss in ops_output_var_3d_mat:   ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],rootgrp.variables[ss]._Fillvalue )[comb_ind.T]
    for ss in ops_output_var_2d_mat:   ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],rootgrp.variables[ss]._Fillvalue )[comb_ind.T]
    for ss in ops_output_ind_2d_mat:   ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],-99999 )[comb_ind.T]

    if TnotS:
        for ss in ops_output_var_3d_T_mat:   ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],rootgrp.variables[ss]._Fillvalue )[comb_ind_T.T]
        for ss in ops_output_ind_3d_T_mat: ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],-99999,)[comb_ind_T.T]
        for ss in ops_output_ind_2d_T_mat: ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],-99999,)[comb_ind_T.T]
    else:
        for ss in ops_output_var_3d_S_mat:   ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],rootgrp.variables[ss]._Fillvalue )[comb_ind_S.T]
        for ss in ops_output_ind_3d_S_mat: ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],-99999,)[comb_ind_S.T]
        for ss in ops_output_ind_2d_S_mat: ops_output_dict[ss] = np.ma.masked_equal(rootgrp.variables[ss],-99999,)[comb_ind_S.T]

    #'pdb.set_trace()
    # need to mask['POTM_IOBSI', 'POTM_IOBSJ','PSAL_IOBSI', 'PSAL_IOBSJ']

    JULD_REFERENCE = datetime.strptime(str(chartostring(rootgrp.variables['JULD_REFERENCE'][:])),'%Y%m%d%H%M%S')
    #pdb.set_trace()

    ops_output_dict['JULD_datetime'] = np.array([JULD_REFERENCE + timedelta(ss) for ss in rootgrp.variables['JULD'][:][comb_ind].ravel()])

    #ops_output_dict['STATION_TYPE'] = stat_type[comb_ind.any(axis = 0)]
    ops_output_dict['STATION_TYPE'] = stat_type[comb_ind]
    '''
    #old slower method
    STATION_IDENTIFIER = []
    for ss, tmperr in zip(rootgrp.variables['STATION_IDENTIFIER'],comb_ind):
        if tmperr:STATION_IDENTIFIER.append(str(chartostring(ss)))
    ops_output_dict['STATION_IDENTIFIER'] = np.array(STATION_IDENTIFIER)
    '''
    ops_output_dict['STATION_IDENTIFIER'] = np.array([str(ss) for ss in chartostring(rootgrp.variables['STATION_IDENTIFIER'][comb_ind,:])])


    #for ss in ops_output_dict.keys(): ss, ops_output_dict[ss].shape

    #for ss in ops_output_dict.keys(): ops_output_dict[ss] = ops_output_dict[ss].ravel()
    #pdb.set_trace()


    rootgrp.close()

    #pdb.set_trace()
    #  for ss in ops_output_dict.keys(): ss, ops_output_dict[ss].shape

    '''
    outputvar_ravel = ['LONGITUDE','LATITUDE','JULD','POSITION_QC','JULD_QC','POTM_IOBSI','POTM_IOBSJ','PSAL_IOBSI','PSAL_IOBSJ','STATION_TYPE']
    outputvar_ravel_T = ['LONGITUDE','LATITUDE','JULD','POSITION_QC','JULD_QC','POTM_IOBSI','POTM_IOBSJ','STATION_TYPE']
    outputvar_ravel_S = ['LONGITUDE','LATITUDE','JULD','POSITION_QC','JULD_QC','PSAL_IOBSI','PSAL_IOBSJ','STATION_TYPE']

    if TnotS:
        outputvar_ravel = outputvar_ravel_T
    else:
        outputvar_ravel = outputvar_ravel_S

    for ss in outputvar_ravel: ops_output_dict[ss] = ops_output_dict[ss].ravel()
    ops_output_dict['DEPTH_QC'] = ops_output_dict['DEPTH_QC'][0,:,:]

    '''


    if TnotS:
        ops_output_dict['OBS'] = ops_output_dict['POTM_OBS'] 
        ops_output_dict['MOD_HX'] = ops_output_dict['POTM_Hx'] 

    else:
            
        ops_output_dict['OBS'] = ops_output_dict['PSAL_OBS'] 
        ops_output_dict['MOD_HX'] = ops_output_dict['PSAL_Hx']


    return ops_output_dict



def obs_reset_sel(Dataset_lst, Fill = True): #,reset_datstr = None):


    obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel = {},{},{},{},{}
    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel = {},{},{}

    for tmp_datstr in Dataset_lst:
        #if reset_datstr is not None:
        #    if tmp_datstr != reset_datstr: continue
        obs_z_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
        obs_obs_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
        obs_mod_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
        obs_lon_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
        obs_lat_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked

        obs_stat_id_sel[tmp_datstr] = ''
        obs_stat_type_sel[tmp_datstr] = None
        obs_stat_time_sel[tmp_datstr] = ''

    return obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel


def profile_line(xlim,ylim,nint = 2000,ni = 375,plotting = False):

    # Create a cross-section between two points, containing only N/S/E/W segements.
    # Also give the direction of each segment, using the DIA/diadct.F90 convention (0:E,1:W,2:S,3:N)
    #   This is one shorter than the the points. it is zero padded at the end.
    #
    #                       Jonathan Tinker 02/09/2018
    #
    # ni = number of lat, perhaps should be nj. i.e.
    # AMM7: ni = 375
    # AMM15: ni = 375
    # ORCA25

    #plotting = False


    if plotting: plt.plot( xlim,ylim,'r')

    # First, simply interpolate 100 points between the line, and round them to integers - this is a simple first guess
    #   however, it will also include the occasional diagonal

    #i_int, j_int = np.linspace(xlim[0],xlim[1],nint),np.linspace(ylim[0],ylim[1],nint)
    #i_int, j_int = np.linspace(xlim[0],xlim[1],nint).astype('int'),np.linspace(ylim[0],ylim[1],nint).astype('int')
    i_int, j_int = np.linspace(xlim[0],xlim[1],nint).round().astype('int'),np.linspace(ylim[0],ylim[1],nint).round().astype('int')

    if plotting: plt.plot( i_int, j_int,'b.-')

    #Convert the integer indexes into a single value
    ij_int = i_int*ni + j_int

    if plotting: plt.plot(ij_int//ni,ij_int%ni, 'rx-')



    # initialise the output directions, and point array (with the first point)
    #dirn = []
    ij_out = []
    ij_out.append(ij_int[0])



    # Loop throug points.
    for ij in ij_int[1:]:
        #note the previous point, and different between it
        pij = ij_out[-1]
        dij = ij-ij_out[-1]


        # If there is no different, skip the point
        if dij in [0]:
            continue
        # If the difference is not horizontal or vertical, add a point (and direction)
        # to fill the gap first
        if dij in [ni-1,ni+1,-ni-1, -ni+1]:
            if dij in [ni-1,ni+1]:
                ij_out.append(pij+ni)
                #dirn.append(1)# West
            elif dij in [-ni-1, -ni+1]:
                ij_out.append(pij-ni)
                #dirn.append(0)# East
            else:
                print('should never stop here')
                pdb.set_trace()
        # if the dij is vertical or horizontal (or diagonal if added the point)
        # add the point
        if  dij in [ni-1,ni+1,-ni-1, -ni+1,ni,-ni,1,-1]:

            ij_out.append(ij)

            #if   dij ==   1: dirn.append(3)# North
            #elif dij ==  -1: dirn.append(2)# South
            #elif dij ==  ni: dirn.append(1)# West
            #elif dij == -ni: dirn.append(0)# East
            #else:
            #    print 'incorrect dir!'


    # convert the 1d index back to 2d indexes
    i_out,j_out = np.array(ij_out)//ni, np.array(ij_out)%ni


    # increment the direction with a zero at the end.
    #dirn.append(0)


    if plotting: plt.plot(i_out,j_out,'ms-')
    if plotting: plt.show()


    #print len(i_out),len(dirn)


    #exterally_calc_dirn = create_dir_from_path(i_out,j_out)


    #pdb.set_trace()
    return i_out,j_out#, exterally_calc_dirn


def pop_up_opt_window(opt_but_names,opt_but_sw = None):
    '''
    ##example inputs
    

    opt_but_names = ['ProfT','SST_ins','SST_sat','ProfS','SLA','ChlA','Hide_Obs','Edges','Loc','Close']
                        
                        
    # button switches  
    opt_but_sw = {}
    opt_but_sw['Hide_Obs'] = {'v':Obs_hide, 'T':'Show Obs','F': 'Hide Obs'}
    opt_but_sw['Edges'] = {'v':Obs_hide, 'T':'Show Edges','F': 'Hide Edges'}
    opt_but_sw['Loc'] = {'v':Obs_hide, 'T':"Don't Selected point",'F': 'Move Selected point'}
    for ob_var in Obs_var_lst_sub:  opt_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var + ' hidden'}
    for ob_var in Obs_varlst:  opt_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var + ' hidden'}


    example use:

    obbut_sel = pop_up_opt_window(opt_but_names, opt_but_sw = opt_but_sw)


    # Set the main figure and axis to be current
    plt.figure(fig.figure)
    plt.sca(clickax)


    # if the button closed was one of the Obs types, add or remove from the hide list
    for ob_var in ['ProfT','SST_ins','SST_sat','ProfS','SLA','ChlA']:
        if obbut_sel == ob_var:
            Obs_vis_d['visible'][ob_var] = not Obs_vis_d['visible'][ob_var] 
    # if the button closed was one of the Obs types, add or remove from the hide list
                
    if obbut_sel == 'Hide_Obs': Obs_hide = not Obs_hide
    if obbut_sel == 'Edges':    Obs_hide_edges = not Obs_hide_edges
    if obbut_sel == 'Loc':      Obs_pair_loc = not Obs_pair_loc



    '''   


    # create Obs options figure
    figobsopt = plt.figure()
    figobsopt.set_figheight(2.5)
    figobsopt.set_figwidth(6)
    
    #add full screen axes
    obcax = figobsopt.add_axes([0,0,1,1], frameon=False)
    
    # Add buttons

    #Create list of dictionaries with the name, and x and y ranges
    obsobbox_l = []
    for opt_i,opt_ss in enumerate(opt_but_names):
        #tmp_obsx0 = (( opt_i*0.2 + 0.1 )//0.8)*0.45 + 0.05
        #tmp_obsdx = 0.4
        tmp_obsx0 = (( opt_i*0.2 + 0.1 )//0.8)*(0.3 + 0.025) + 0.025
        tmp_obsdx = 0.3
        #tmp_obsy0 = (( opt_i*0.2 + 0.1 )%0.8)
        #tmp_obsdy = 0.15
        #tmp_obsy0 = 1-(( opt_i*0.2 + 0.1 )%0.8)
        #tmp_obsdy = -0.15
        #tmp_obsy0 = 1-(( opt_i*0.16 + 0.04 )%0.8)
        tmp_obsdy = -0.2
        tmp_obsy0 = 1-(( opt_i*0.24 + 0.04 )%(1-0.04))

        obsobbox_l.append({'x':np.array([tmp_obsx0,tmp_obsx0+tmp_obsdx]),'y':np.array([tmp_obsy0,tmp_obsy0+tmp_obsdy]),'name':opt_ss})


    # draw buttons
    for tmpoob in obsobbox_l: obcax.plot(tmpoob['x'][[0,1,1,0,0]],tmpoob['y'][[0,0,1,1,0]],'k')

    obcax_tx_hd = {}
    # write button names 
    for tmpoob in obsobbox_l: 
        obcax_tx_hd[tmpoob['name']] = obcax.text(tmpoob['x'].mean(),tmpoob['y'].mean(),tmpoob['name'],ha = 'center', va = 'center')
   

    # if a swich is provided, change button titles based on T/F in switch
    if opt_but_sw is not None:
        for tmpoob in opt_but_sw.keys():
            if isinstance(opt_but_sw[tmpoob]['v'],bool):
                if opt_but_sw[tmpoob]['v']:
                    obcax_tx_hd[tmpoob].set_text(opt_but_sw[tmpoob]['T'])
                    if 'T_col' in opt_but_sw[tmpoob].keys():obcax_tx_hd[tmpoob].set_color(opt_but_sw[tmpoob]['T_col'])
                else:
                    obcax_tx_hd[tmpoob].set_text(opt_but_sw[tmpoob]['F'])
                    if 'F_col' in opt_but_sw[tmpoob].keys():obcax_tx_hd[tmpoob].set_color(opt_but_sw[tmpoob]['F_col'])
            elif isinstance(opt_but_sw[tmpoob]['v'],(int, float)):
                obcax_tx_hd[tmpoob].set_text(opt_but_sw[tmpoob][int(opt_but_sw[tmpoob]['v'])])

    # Set x and y lims
    obcax.set_xlim(0,1)
    obcax.set_ylim(0,1)
    
    # redraw canvas
    figobsopt.canvas.draw()
    
    #flush canvas
    figobsopt.canvas.flush_events()
    
    # Show plot, and set it as the current figure and axis
    figobsopt.show()
    plt.figure(figobsopt.figure)
    plt.sca(obcax)


    # await button press...
    #   keep trying until close_obcax is True
    close_obcax = False
    while close_obcax == False:

        # get click location
        tmpobsbutloc = plt.ginput(1, timeout = 3) #[(0.3078781362007169, 0.19398809523809524)]

        if len(tmpobsbutloc)!=1:
            #print('tmpobsbutloc len != 1',tmpobsbutloc )
            continue
            pdb.set_trace()
        else:
            if len(tmpobsbutloc[0])!=2:
                #print('tmpobsbutloc[0] len != 2',tmpobsbutloc )
                continue
                pdb.set_trace()
        # was a button clicked?
        obbut = []

        # cycle through the buttons, and ask if the click was within there x and y lims
        for tmpoob in obsobbox_l: 

            # if so, record which and allow the window to close
            if (tmpobsbutloc[0][0] >= tmpoob['x'].min()) & (tmpobsbutloc[0][0] <= tmpoob['x'].max()) & (tmpobsbutloc[0][1] >= tmpoob['y'].min()) & (tmpobsbutloc[0][1] <= tmpoob['y'].max()):
                obbut.append(True)

                close_obcax = True
            else:
                obbut.append(False)
                
        # find which button was closed.         
        obbut_mat = np.array(obbut)
        if obbut_mat.any():
            obbut_sel = obsobbox_l[np.where(obbut_mat)[0][0]]['name']
            print('obbut_sel:',obbut_sel)
        else:
            obbut_sel = ''
            close_obcax = False
            
        # quit of option box is closed without button press.
        if plt.fignum_exists(figobsopt) == False:
            close_obcax = True
            
    # close figure
    if figobsopt is not None:
        if plt.fignum_exists(figobsopt.number):
            plt.close(figobsopt)

    return obbut_sel

def pop_up_info_window(help_text): #obs_but_names,obs_but_sw = None
  
    #pdb.set_trace()

    # create Obs options figure
    fighelp = plt.figure()
    fighelp.set_figheight(6)
    fighelp.set_figwidth(6)
    
    #add full screen axes
    #helpax = fighelp.add_axes([0.05,0.05,0.95,0.95], frameon=False)
    helpax = fighelp.add_axes([0,0,1,1], frameon=False)
  
    # Set x and y lims
    helpax.set_xlim(0,1)
    helpax.set_ylim(0,1)

    helpax.text(0.05,0.95,help_text, ha= 'left', va = 'top', wrap = True)
    
    # redraw canvas
    fighelp.canvas.draw()
    
    #flush canvas
    fighelp.canvas.flush_events()
    
    # Show plot, and set it as the current figure and axis
    fighelp.show()
    plt.figure(fighelp.figure)
    plt.sca(helpax)


    close_helpax = False
    while close_helpax == False:

        # get click location
        tmphelpbutloc = plt.ginput(1, timeout = 3) #[(0.3078781362007169, 0.19398809523809524)]

        if len(tmphelpbutloc)!=1:
            print('tmphelpbutloc len != 1',tmphelpbutloc )
            continue
            pdb.set_trace()
        else:
            if len(tmphelpbutloc[0])!=2:
                print('tmphelpbutloc[0] len != 2',tmphelpbutloc )
                continue
                pdb.set_trace()
            # was a button clicked?
            # if so, record which and allow the window to close
            if (tmphelpbutloc[0][0] >= 0) & (tmphelpbutloc[0][0] <= 1) & (tmphelpbutloc[0][1] >= 0) & (tmphelpbutloc[0][1] <= 1):
                
                close_helpax = True

        # quit of option box is closed without button press.
        if plt.fignum_exists(fighelp) == False:
            close_helpax = True
            

    # close figure
    if close_helpax:
        if fighelp is not None:
            if plt.fignum_exists(fighelp.number):
                plt.close(fighelp)



def get_help_text(help_type,help_but):
    help_text = 'Help: %s\n===================================\n\n'%help_but
    help_text= help_text + 'When clicking an axes, you select a new point, depth or time, depending on the axis selected.\n\n'
    help_text= help_text + 'When a variable button is clicked (on the left hand side) you change the current varaible.\n'
    help_text= help_text + 'When a function button clicked (on the right hand side) a function is executed.\n'
    help_text= help_text + 'Buttons with a double outline behave differently when right clicked.\n'
        
    help_text = help_text + '\n\n'


    if help_type.lower() == 'axis':
        help_text= help_text + 'Axis selected:\n\n'
        if help_but == 'axis: a':
            help_text= help_text + 'The main axis (a) changes the selected location (latitude and longitude).\n'
        elif help_but == 'axis: b':
            help_text= help_text + 'The main axis (b) changes the selected longitude.\n'
        elif help_but == 'axis: c':
            help_text= help_text + 'The main axis (c) changes the selected latitude.\n'
        elif help_but == 'axis: d':
            help_text= help_text + 'The main axis (d) changes the selected depth.\n'
        elif help_but == 'axis: e':
            help_text= help_text + 'The main axis (e) changes the selected time.\n'
        elif help_but == 'axis: f':
            help_text= help_text + 'The main axis (f) changes the selected depth.\n'   


    elif help_type.lower() == 'var':
        help_text= help_text + 'Variable selected: %s\n\n'%help_but       


    elif help_type.lower() == 'func':
        help_text= help_text + 'Function selected: %s\n\n'%help_but
        if help_but == 'Hov/Time':
            help_text = help_text + 'Shows or hides the Hovmoller/Time axis (d and e) - hiding this data means the time data is '
            help_text = help_text + 'not loaded, which is quicker.'
        elif help_but == 'Show Prof':
            help_text = help_text + 'Shows or hides the profile axis (f), adjusting the size of the cross-section panels.'
        elif help_but == 'Zoom':
            help_text = help_text + 'Zoom in or out or reset (for left or right or middle button click), or setting max depth.\n'
            help_text = help_text + 'If Zoom is clicked with the MIDDLE button:\n'
            help_text = help_text + 'The zoom is reset.\n'
            help_text = help_text + 'If Zoom is clicked with the LEFT button:\n'
            help_text = help_text + 'Click on the main axis (a) twice, to deliniate a square that will then be zoomed into.\n'
            help_text = help_text + 'If Zoom is clicked with the RIGHT button:\n'
            help_text = help_text + 'Click on the main axis (a) twice, to deliniate a square that will then be zoomed out from - '
            help_text = help_text + 'i.e. the corners of the current map will be rescaled to the clicked points. The closer to the '
            help_text = help_text + 'centre you click, more agressive the zoom out will be.\n'
            help_text = help_text + 'or, click on the axis b or c once and the maximum depth will be set.\n'
            help_text = help_text + 'When you click once, the zoom button will turn red (or green for right click), '
            help_text = help_text + 'when you click twice, it will change back to black.'
        #elif help_but == 'Reset zoom':
        #    help_text = help_text + 'Reset the Zoom to default.'
        elif help_but == 'ColScl':
            help_text = help_text + 'Changes the colourmap scale, from linear, to focusing on the high or lower values. '
            help_text = help_text + 'Clicking on this cycles through these options, which is reflected in the button label '
            help_text = help_text + '(Col: Linear Col: High, Col: Low).'
        elif help_but == 'Axis':
            help_text = help_text + 'Changes the x- and y-axis scaling for the map axis (a) between Axis: Auto and Axis: Equal '
            help_text = help_text + '(with button labels updating) where Axis: Auto maximises the x and y ranges, '
            help_text = help_text + 'which can distort the map image, and Axis: Equal where one 1 degree of latitude is '
            help_text = help_text + 'the same size as 1 degree of longitude, which is also a distortion, but sometime less so.'
            help_text = help_text + ''
        #elif help_but == 'Clim: Reset':
        #    help_text = help_text + 'Resets the colour map limits to the default (showing the 5th and 95th percentile values of the image).'
        #    help_text = help_text + ''
        #    help_text = help_text + ''
        elif help_but == 'Clim: Zoom (zooms in, expands or resets colourmap limits for left, right and middle click)':
            help_text = help_text + 'If Clim: Zoom is clicked with the LEFT button:\n'
            help_text = help_text + 'Allows the user to zoom the colormap limits, with two click on the axis a colormap, '
            help_text = help_text + 'one for the desired colourmap minima and one for the desired colourmap maxima.'
            help_text = help_text + 'If Clim: Zoom is clicked with the RIGHT button:\n'
            help_text = help_text + 'Expands the colourmap limit, increasing both the minima and maxima by 50% of the range, '
            help_text = help_text + 'this allows the user to select the colourmap limits outside those shown.'
            help_text = help_text + 'If Clim: Zoom is clicked with the MIDDLE button:\n'
            help_text = help_text + 'Resets the colour map limits to the default (showing the 5th and 95th percentile values of the image).'
            help_text = help_text + ''
        #elif help_but == 'Clim: Expand':
        #    help_text = help_text + 'Expands the colourmap limit, increasing both the minima and maxima by 50% of the range, '
        #    help_text = help_text + 'this allows the user to select the colourmap limits outside those shown.'
        #    help_text = help_text + ''
        elif help_but == 'Clim: pair':
            help_text = help_text + 'When two different datasets are shown, their colourmap limits are optimised for the current view '
            help_text = help_text + 'for each dataset. Clim: pair links these together, so the two dataset can be compared visually.'
            help_text = help_text + ''
        elif help_but == 'Clim: sym':
            help_text = help_text + 'Makes the colourmap symetrical about zero, and uses a blue - white - red colourmap (matplotlib seismic).'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but in ['Surface','Near-Bed','Surface-Bed','Depth-Mean','Depth level']:            
            help_text = help_text + 'Surface, Near-Bed, Surface-Bed, Depth-Mean and Depth level '
            help_text = help_text + 'sets the depth level or processing shown in the map axis (a). The current option is colour red. ' 
            help_text = help_text + 'Clicking on axis d changes the depth, and changes the mode to Depth Level.'
        elif help_but == 'Contours':
            help_text = help_text + 'Shows or hides contours based on the colourbar tick values.'
        elif help_but == 'Grad':
            help_text = help_text + 'Gradient of the data, either horizontal or vertical, with options.'
            help_text = help_text + 'Left and right clicks cycles between off (greyed out grad), to the horizontal gradient (Grad: Horiz) and the vertical Gradient (Grad: Vert). '
            help_text = help_text + 'Left clicking moves forward through the sequence (No Grad, Horiz Grad, Vert Grad), right click moves throught the sequence backwards. '
            help_text = help_text + 'Central Click displays options window. These include:\n'
            help_text = help_text + ' - Grad Meth: Centred Diff, or Forward Diff - default is centred difference, but forward differnce is better to show gridscale noise.\n'
            help_text = help_text + ' - Grad 2D Method: magnitude; d/dx; d/dy. For the main map (a); the spatial gradient can be displayed as the magnitude, or the eastward or northward component\n'
            help_text = help_text + ' - Pre-proc: |x| ; x. The absolute of the data can be taken before calculating the gradient (default: off (x)).\n'
            help_text = help_text + ' - Post-proc: |x| ; x. The absolute of the gradient can be taken before after calculating it, before displaying it (default: off (x)).\n'
            help_text = help_text + ' - dx(dy/dx) | dy/dx. The gradient can be multiplied by the spacing, to effectively give the difference between grid boxes.\n'
            #help_text = help_text + ' - grad_regrid_xy (not coded). Option to regrid the lon/lat, as the forward differnece option shifts the data.\n'
        elif help_but == 'T Diff':
            help_text = help_text + 'Shows the difference between the current time and the previous time, '
            help_text = help_text + 'i.e. how much it has change since the previous day etc. Greyed out if the first time of the dataset is selected.'
            help_text = help_text + ''
        elif help_but == 'TS Diag':
            help_text = help_text + 'Produces a Temperature Salinity diagram for the selected point. '
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'LD time':
            help_text = help_text + 'Cycles through lead times if a series of forecasts are loaded.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'Fcst Diag':
            help_text = help_text + 'Produces a Forecast diagnostic spaghetti diagram in a separate window.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'Vis curr':
            help_text = help_text + 'Show the current field with "current barbs", akin to windbarbs. If 3d currents are available, '
            help_text = help_text + 'the currents are processed the same way as the field shown in the map axes (a), i.e. the surface '
            help_text = help_text + 'currents, the depth slice, the near bottom currents etc. '
        elif help_but == 'Obs':
            help_text = help_text + 'Compare the model to the observations. \n'
            help_text = help_text + '1) After clicking on the Obs button, you can click the map to select an obserations, and it will be displayed on the axes f, the profile window, or\n'
            help_text = help_text + '2) you can right click on the Obs button, and an option window will open, where you can select which obseration type to show, '
            help_text = help_text + 'whether you want show or hide the observations, or their edges.'
        elif help_but == 'Xsect':
            help_text = help_text + 'Plot a user defined cross section. \n'
            help_text = help_text + 'The first click, or subsequent right click allow you to select a cross-section, which is then plotted. Subsequent (left) click '
            help_text = help_text + 'plots the selected cross section of current variable and time. If two datasets, shows both, and their difference, otherwise shows '
            help_text = help_text + 'the current dataset. The cross-section window will close when it is clicked.\n\n'
            help_text = help_text + 'Selecting a cross-section\n'
            help_text = help_text + '---------------------------\n'
            help_text = help_text + 'A cross section is selected with ginput(-1) by left clicking on the map. Each point will appear as a red cross. the last point can be '
            help_text = help_text + 'removed with a right click. When the desired cross section is selected, middle click to exit, and plot. '
        elif help_but == 'Save Figure':
            help_text = help_text + 'Saves a png of the displayed view (excluding the buttons), with a text file containing the current options to allow '
            help_text = help_text + 'the view to be recreated in a batch mode with the just plot options.'
            help_text = help_text + ''
        elif help_but == 'Help':
            help_text = help_text + 'Displays help on the selected option.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'Quit':
            help_text = help_text + 'Quits the programme.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'Sec Grid':
            help_text = help_text + 'When two data sets are from differnt configurations, the second (etc) are interpolated onto the grid of the first config. '
            help_text = help_text + 'This allows you to display the secondary models on there native grids. When showing difference plots, they are still regridded onto the first model grid.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but == 'MLD':
            help_text = help_text + 'Shows the mixed layer depth on the cross section and hovmuller subplots (b, c and d). '
            help_text = help_text + 'Right clicking gives a window where you can select which MLD variable within the file you would like to plot.'
            help_text = help_text + ''
            help_text = help_text + ''
        elif help_but in ['Click','Loop']:
            help_text = help_text + 'Changes the mode, between Click and Loop, with the current mode highlighted in yellow. '
            help_text = help_text + 'Click mode is the normal mode where the program is waiting for the user to click on a '
            help_text = help_text + 'button or an axis. Loop automatically cycles through the times until Click mode is reactivated.\n\n'
            help_text = help_text + 'To reactivated click mode, point the button in the click button and wait for the view to stop cycling '
            help_text = help_text + 'before clicking Click. '
        elif help_but.split(' ')[0] == 'Dataset':
            help_text = help_text + 'Change between Datasets.'
        elif help_but.split('-')[0] == 'Dat':
            help_text = help_text + 'The difference between Datasets.'
        elif help_but == 'regrid_meth':
            help_text = help_text + 'Changes the regridding method when different configurations are compared. '
            help_text = help_text + 'Cycles through Regrid: Bilin for bilinear interpolation and Regrid: NN for nearest neighbour interpolation.'

    help_text = help_text + '\n\nClick on this window to close.'

    return help_text

def jjii_from_lon_lat(lon_in, lat_in, lon_d_in,lat_d_in,config = 'orca12'):
    
    tmp_distmat = np.sqrt((lon_in - lon_d_in)**2 + (lat_in - lat_d_in)**2)


    sel_jj,sel_ii = tmp_distmat.argmin()//tmp_distmat.shape[1], tmp_distmat.argmin()%tmp_distmat.shape[1]

    return sel_jj,sel_ii
               
def calc_ens_stat_3d(ns_slice_dat, ew_slice_dat,hov_dat,ns_slice_dict,ew_slice_dict,hov_dat_dict,ts_dat_dict, Ens_stat,Dataset_lst):

    #pdb.set_trace()
    if Ens_stat is None:
        ens_ns_slice_dat, ens_ew_slice_dat,ens_hov_dat = ns_slice_dat, ew_slice_dat,hov_dat
    elif Ens_stat == 'EnsMean':
        ens_ns_slice_dat = np.ma.array([ns_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mean(axis = 0)
        ens_ew_slice_dat = np.ma.array([ew_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mean(axis = 0)
        ens_hov_dat = np.ma.array([hov_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mean(axis = 0)
    elif Ens_stat == 'EnsVar':
        ens_ns_slice_dat = np.ma.array([ns_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).var(axis = 0)
        ens_ew_slice_dat = np.ma.array([ew_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).var(axis = 0)
        ens_hov_dat = np.ma.array([hov_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).var(axis = 0)
    elif Ens_stat == 'EnsStd':
        ens_ns_slice_dat = np.ma.array([ns_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).std(axis = 0)
        ens_ew_slice_dat = np.ma.array([ew_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).std(axis = 0)
        ens_hov_dat = np.ma.array([hov_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).std(axis = 0)
    elif Ens_stat == 'EnsCnt':
        ens_ns_slice_dat = (np.ma.array([ns_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mask==False).sum(axis = 0)
        ens_ew_slice_dat = (np.ma.array([ew_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mask==False).sum(axis = 0)
        ens_hov_dat = (np.ma.array([hov_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mask==False).sum(axis = 0)
    else:
        pdb.set_trace()


    ts_dat_dat = np.ma.array([ts_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ])

    ens_ts_dat = np.ma.array((ts_dat_dat.mean(axis = 0)-2*ts_dat_dat.std(axis = 0),
                                    ts_dat_dat.mean(axis = 0),
                                    ts_dat_dat.mean(axis = 0)+2*ts_dat_dat.std(axis = 0)))
    
    return  ens_ns_slice_dat, ens_ew_slice_dat,ens_hov_dat, ens_ts_dat

#def calc_ens_stat_2d_temp(ns_slice_dict,ew_slice_dict,hov_dat_dict,ts_dat_dict, Ens_stat,Dataset_lst):
#    pdb.set_trace()

def calc_ens_stat_2d(ns_slice_dict,ew_slice_dict,ts_dat_dict, Ens_stat,Dataset_lst):



    ens_ns_slice_dat = np.ma.array([ns_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ])
    ens_ew_slice_dat = np.ma.array([ew_slice_dict[tmpdatstr] for tmpdatstr in Dataset_lst ])
    ts_dat_dat = np.ma.array([ts_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ])


    ens_ns_slice_dat = np.ma.array((ens_ns_slice_dat.mean(axis = 0)-2*ens_ns_slice_dat.std(axis = 0),
                                    ens_ns_slice_dat.mean(axis = 0),
                                    ens_ns_slice_dat.mean(axis = 0)+2*ens_ns_slice_dat.std(axis = 0)))
    
    ens_ew_slice_dat = np.ma.array((ens_ew_slice_dat.mean(axis = 0)-2*ens_ew_slice_dat.std(axis = 0),
                                    ens_ew_slice_dat.mean(axis = 0),
                                    ens_ew_slice_dat.mean(axis = 0)+2*ens_ew_slice_dat.std(axis = 0)))
    
    ens_ts_dat = np.ma.array((ts_dat_dat.mean(axis = 0)-2*ts_dat_dat.std(axis = 0),
                                    ts_dat_dat.mean(axis = 0),
                                    ts_dat_dat.mean(axis = 0)+2*ts_dat_dat.std(axis = 0)))
    
        
    return  ens_ns_slice_dat, ens_ew_slice_dat, ens_ts_dat

def calc_ens_stat_map(map_dat_dict, Ens_stat,Dataset_lst):

    if Ens_stat == 'EnsMean':
        map_dat = np.ma.array([map_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mean(axis = 0)
    elif Ens_stat == 'EnsVar':
        map_dat = np.ma.array([map_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).var(axis = 0)
    elif Ens_stat == 'EnsStd':
        map_dat = np.ma.array([map_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).std(axis = 0)
    elif Ens_stat == 'EnsCnt':
        map_dat = (np.ma.array([map_dat_dict[tmpdatstr] for tmpdatstr in Dataset_lst ]).mask==False).sum(axis = 0)

    return map_dat


def load_xypos(xypos_fname):
    #xypos_dict = {}
    #xypos_dict[tmp_datstr] =  load_xypos(config_fnames_dict[tmpconfig]['xypos_file'])
    #xypos_dict = load_xypos(xypos_fname)
            


    from scipy.interpolate import griddata


    xypos_dict = {}

    xypos_dict['do_xypos'] = True
    rootgrp = Dataset(xypos_fname, 'r')
    for xy_var in rootgrp.variables.keys(): xypos_dict[xy_var] = rootgrp.variables[xy_var][:]
    xypos_dict['lon_min'] = xypos_dict['LON'].min()
    xypos_dict['lat_min'] = xypos_dict['LAT'].min()
    xypos_dict['dlon'] =  (np.diff(xypos_dict['LON'][0,:])).mean()
    xypos_dict['dlat'] =  (np.diff(xypos_dict['LAT'][:,0])).mean()
    
    rootgrp.close()



    nxylat, nxylon = xypos_dict['LAT'].shape
    xypos_mask =  np.ma.getmaskarray(xypos_dict['XPOS'])

    xypos_xmat, xypos_ymat = np.meshgrid(np.arange(nxylon), np.arange(nxylat))

    points = (xypos_xmat[~xypos_mask], xypos_ymat[~xypos_mask])
    values_X = xypos_dict['XPOS'][~xypos_mask]
    values_Y = xypos_dict['YPOS'][~xypos_mask]

    #plt.plot(points[0],points[1],'x')
    #plt.show()
    #pdb.set_trace()
    
    xypos_dict['XPOS_NN'] = griddata(points, values_X, (xypos_xmat, xypos_ymat), method='nearest')
    xypos_dict['YPOS_NN'] = griddata(points, values_Y, (xypos_xmat, xypos_ymat), method='nearest')

    return xypos_dict
    

def cut_down_xypos(xypos_dict, lonmin, latmin,lonmax, latmax):
    


    '''
    (Pdb) for ss in ['LON', 'LAT', 'XPOS', 'YPOS', 'lon_min', 'lat_min', 'dlon', 'dlat', 'XPOS_NN', 'YPOS_NN']:ss,xypos_dict[ss].shape
    ('LON', (1801, 3601))
    ('LAT', (1801, 3601))
    ('XPOS', (1801, 3601))
    ('YPOS', (1801, 3601))
    ('lon_min', ())
    ('lat_min', ())
    ('dlon', ())
    ('dlat', ())
    ('XPOS_NN', (1801, 3601))
    ('YPOS_NN', (1801, 3601))

    
    '''

    x0 = np.abs(xypos_dict['LON'][0,:]  - lonmin).argmin()-1
    y0 = np.abs(xypos_dict['LAT'][:,0] - latmin).argmin()-1
    x1 = np.abs(xypos_dict['LON'][0,:] - lonmax).argmin()+1
    y1 = np.abs(xypos_dict['LAT'][:,0]  - latmax).argmin()+1

    #pdb.set_trace()

    reduced_xypos_dict = {}
    
    reduced_xypos_dict['LON'] = xypos_dict['LON'][y0:y1+1,x0:x1+1]
    reduced_xypos_dict['LAT'] = xypos_dict['LAT'][y0:y1+1,x0:x1+1]
    reduced_xypos_dict['XPOS'] = xypos_dict['XPOS'][y0:y1+1,x0:x1+1]
    reduced_xypos_dict['YPOS'] = xypos_dict['YPOS'][y0:y1+1,x0:x1+1]
    reduced_xypos_dict['XPOS_NN'] = xypos_dict['XPOS_NN'][y0:y1+1,x0:x1+1]
    reduced_xypos_dict['dlon'] = xypos_dict['dlon']
    reduced_xypos_dict['dlat'] = xypos_dict['dlat']
    reduced_xypos_dict['lon_min'] = reduced_xypos_dict['LON'].min()
    reduced_xypos_dict['lat_min'] = reduced_xypos_dict ['LAT'].min()
    reduced_xypos_dict['do_xypos'] = xypos_dict['do_xypos']


    #pdb.set_trace()

    return reduced_xypos_dict

def int_ind_wgt_from_xypos(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,loni,latj):


    #Numeric code of the dataset
    th_d_ind = int(tmp_datstr[8:])
    tmp_xypos_dict = xypos_dict[tmp_datstr]
    tmp_thd = thd[th_d_ind]
    tmp_thd_x0 = tmp_thd['x0']
    tmp_thd_y0 = tmp_thd['y0']
    tmp_thd_dx = tmp_thd['dx']
    tmp_thd_dy = tmp_thd['dy']
    tmp_thd_cutx0 = tmp_thd['cutx0']
    tmp_thd_cuty0 = tmp_thd['cuty0']
    #tmp_lat_d_2 = lat_d[2]
    #tmp_lon_d_2 = lon_d[2]
    tmp_lat_d_2 = lat_d[th_d_ind]
    tmp_lon_d_2 = lon_d[th_d_ind]
    #pdb.set_trace()

    sel_bl_jj_out, sel_bl_ii_out, NWS_wgt, sel_jj_out, sel_ii_out = int_ind_wgt_from_xypos_func(tmp_xypos_dict,
        loni,latj, tmp_lon_d_2,tmp_lat_d_2, tmp_thd_x0 = tmp_thd_x0,tmp_thd_y0 = tmp_thd_y0,tmp_thd_dx = tmp_thd_dx,tmp_thd_dy = tmp_thd_dy,tmp_thd_cutx0=tmp_thd_cutx0, tmp_thd_cuty0=tmp_thd_cuty0)

    return sel_bl_jj_out, sel_bl_ii_out, NWS_wgt, sel_jj_out, sel_ii_out

    
def int_ind_wgt_from_xypos_func(tmp_xypos_dict, loni,latj, tmp_lon_d_2,tmp_lat_d_2, tmp_thd_x0 = 0,tmp_thd_y0 = 0,tmp_thd_dx = 1,tmp_thd_dy = 1,tmp_thd_cutx0=0, tmp_thd_cuty0=0):
    '''
    #sel_bl_jj_out, sel_bl_ii_out, NWS_wgt, sel_jj_out, sel_ii_out = int_ind_wgt_from_xypos_func(tmp_xypos_dict, loni,latj, tmp_lon_d_2,tmp_lat_d_2, tmp_thd_x0 = 0,tmp_thd_y0 = 0,tmp_thd_dx = 1,tmp_thd_dy = 1):
    
    For a given lon and lat (loni, latj), find the ii,jj index of a given dataset.

    If the xypos files are available, use them, otherwise, the method depends on the config.                   
    '''

    


    test_plot = False
 
    #        test_plot = True
    if test_plot:
        plt.plot(loni.ravel(),latj.ravel(),'k.')
        plt.plot(tmp_lon_d_2.ravel(),tmp_lat_d_2.ravel(),'bx')

    # if using xypos files:
    
    #Find size of xypos and config lat arrays. 
    nxylat, nxylon = tmp_xypos_dict['LAT'].shape
    #ncurlat, ncurlon = lat_d[th_d_ind].shape
    ncurlat, ncurlon = tmp_lat_d_2.shape  
    ndatlat, ndatlon = loni.shape

    # find nearest grid box in the xypos arrays using a y = mx + c method
    # first for a given lon/lat array, find the index in the xypos file
    xy_i_ind = ((loni-tmp_xypos_dict['lon_min'])/tmp_xypos_dict['dlon']).astype('int')
    xy_j_ind = ((latj-tmp_xypos_dict['lat_min'])/tmp_xypos_dict['dlat']).astype('int')
    xy_i_ind = np.ma.minimum(np.ma.maximum(xy_i_ind,0),nxylon-1)
    xy_j_ind = np.ma.minimum(np.ma.maximum(xy_j_ind,0),nxylat-1)


    if test_plot:plt.plot(tmp_xypos_dict['LON'][xy_j_ind,xy_i_ind].ravel(), tmp_xypos_dict['LAT'][xy_j_ind,xy_i_ind].ravel(),'rx')

    # having found the index in the xypos file of the nearest grid box
    #   find its i,j for the Dataset 2 lon/lat array
    sel_ii_out = np.floor((tmp_xypos_dict['XPOS'][xy_j_ind,xy_i_ind] - tmp_thd_x0)/tmp_thd_dx).astype('int')
    sel_jj_out = np.floor((tmp_xypos_dict['YPOS'][xy_j_ind,xy_i_ind] - tmp_thd_y0)/tmp_thd_dy).astype('int')

    # As were in the indices of the Dataset 2 lon/lat array, we need to consider cut outs
    sel_ii_out-=tmp_thd_cutx0
    sel_jj_out-=tmp_thd_cuty0
    #
    #ensure its with the domain.
    sel_ii_out = np.ma.minimum(np.ma.maximum(sel_ii_out,0),ncurlon-1)
    sel_jj_out = np.ma.minimum(np.ma.maximum(sel_jj_out,0),ncurlat-1)

    if test_plot:plt.plot(tmp_lon_d_2[sel_jj_out,sel_ii_out].ravel(), tmp_lat_d_2[sel_jj_out,sel_ii_out].ravel(),'c+')


    test_sel_ii_out = sel_ii_out.copy()
    test_sel_jj_out = sel_jj_out.copy()
    


    # find nearest grid box in the xypos arrays using a y = mx + c method
    # first for a given lon/lat array, find the index in the xypos file
    #   treat this an a bilinear float, and as a nn int
    #   (indices in the xypos file)
    
    NWS_flt_ii_ind = ((loni-tmp_xypos_dict['lon_min'])/tmp_xypos_dict['dlon'])#.astype('int')
    NWS_flt_jj_ind = ((latj-tmp_xypos_dict['lat_min'])/tmp_xypos_dict['dlat'])#.astype('int')

    NWS_nn_ii_ind = NWS_flt_ii_ind.astype('int')
    NWS_nn_jj_ind = NWS_flt_jj_ind.astype('int')

    #make an array for the surrounding corners
    NWS_lrbt_dist = np.zeros((4,ndatlat, ndatlon), dtype = 'float')
    NWS_wgt = np.ma.zeros((4,ndatlat, ndatlon), dtype = 'float')

    NWS_bl_ii_ind = np.zeros((4,ndatlat, ndatlon), dtype = 'int')
    NWS_bl_jj_ind = np.zeros((4,ndatlat, ndatlon), dtype = 'int')

    # pick the indices for the surrounding corners
    #   (indices in the xypos file)
    NWS_bl_ii_ind[0,:,:] = np.floor(NWS_flt_ii_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_jj_ind[0,:,:] = np.floor(NWS_flt_jj_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_ii_ind[1,:,:] = np.ceil( NWS_flt_ii_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_jj_ind[1,:,:] = np.floor(NWS_flt_jj_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_ii_ind[2,:,:] = np.floor(NWS_flt_ii_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_jj_ind[2,:,:] = np.ceil( NWS_flt_jj_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_ii_ind[3,:,:] = np.ceil( NWS_flt_ii_ind[:,:]).astype('int') # BL, BR, TL, TR
    NWS_bl_jj_ind[3,:,:] = np.ceil( NWS_flt_jj_ind[:,:]).astype('int') # BL, BR, TL, TR

    # and the distace to them (in degrees within the XYPOS file)
    NWS_lrbt_dist[0,:,:] = (NWS_flt_ii_ind[:,:] - NWS_bl_ii_ind[0,:,:])/(NWS_bl_ii_ind[1,:,:] - NWS_bl_ii_ind[0,:,:]) # distance from left handside
    NWS_lrbt_dist[1,:,:] = (NWS_bl_ii_ind[1,:,:] - NWS_flt_ii_ind[:,:])/(NWS_bl_ii_ind[1,:,:] - NWS_bl_ii_ind[0,:,:]) # distance from right handside
    NWS_lrbt_dist[2,:,:] = (NWS_flt_jj_ind[:,:] - NWS_bl_jj_ind[0,:,:])/(NWS_bl_jj_ind[2,:,:] - NWS_bl_jj_ind[0,:,:]) # distance from bottom handside
    NWS_lrbt_dist[3,:,:] = (NWS_bl_jj_ind[2,:,:] - NWS_flt_jj_ind[:,:])/(NWS_bl_jj_ind[2,:,:] - NWS_bl_jj_ind[0,:,:]) # distance from top handside
    
    #################################################################################
    # to catch when NWS_flt_ii_ind%1 == 0
    #################################################################################
    NWS_lrbt_dist[0,:,:][NWS_bl_ii_ind[1,:,:] == NWS_bl_ii_ind[0,:,:]] = 0.5
    NWS_lrbt_dist[1,:,:][NWS_bl_ii_ind[1,:,:] == NWS_bl_ii_ind[0,:,:]] = 0.5
    NWS_lrbt_dist[2,:,:][NWS_bl_jj_ind[2,:,:] == NWS_bl_jj_ind[0,:,:]] = 0.5
    NWS_lrbt_dist[3,:,:][NWS_bl_jj_ind[2,:,:] == NWS_bl_jj_ind[0,:,:]] = 0.5
    #################################################################################
    
    NWS_wgt[0,:,:] = (NWS_lrbt_dist[1,:,:]*NWS_lrbt_dist[3,:,:]) # BL: dist to TR
    NWS_wgt[1,:,:] = (NWS_lrbt_dist[0,:,:]*NWS_lrbt_dist[3,:,:]) # BR: dist to TL
    NWS_wgt[2,:,:] = (NWS_lrbt_dist[1,:,:]*NWS_lrbt_dist[2,:,:]) # TL: dist to BR
    NWS_wgt[3,:,:] = (NWS_lrbt_dist[0,:,:]*NWS_lrbt_dist[2,:,:]) # TR: dist to BR
 

    '''
    if (np.floor(NWS_flt_ii_ind[:,:]) == np.ceil(NWS_flt_ii_ind[:,:])).any():
        print('NWS_flt_ii_ind%0 == 0')
        pdb.set_trace()
    if (np.floor(NWS_flt_jj_ind[:,:]) == np.ceil(NWS_flt_jj_ind[:,:])).any():
        print('NWS_flt_ii_ind%0 == 0')
        pdb.set_trace()
    '''

    #mask weightings
    NWS_bl_jj_ind_final,NWS_bl_ii_ind_final = np.ma.array(NWS_bl_jj_ind.copy()),np.ma.array(NWS_bl_ii_ind.copy())
    NWS_wgt[:,(NWS_bl_jj_ind_final<0).any(axis = 0)] = np.ma.masked
    NWS_wgt[:,(NWS_bl_ii_ind_final<0).any(axis = 0)] = np.ma.masked
    #NWS_wgt[:,(NWS_bl_jj_ind_final>=ndatlat).any(axis = 0)] = np.ma.masked
    #NWS_wgt[:,(NWS_bl_ii_ind_final>=ndatlon).any(axis = 0)] = np.ma.masked
    #NWS_wgt[:,(NWS_bl_jj_ind_final>=ncurlat).any(axis = 0)] = np.ma.masked
    #NWS_wgt[:,(NWS_bl_ii_ind_final>=ncurlon).any(axis = 0)] = np.ma.masked
    NWS_wgt[:,(NWS_bl_jj_ind_final>=nxylat).any(axis = 0)] = np.ma.masked
    NWS_wgt[:,(NWS_bl_ii_ind_final>=nxylon).any(axis = 0)] = np.ma.masked

    NWS_bl_jj_ind_final[NWS_wgt.mask == True] = 0
    NWS_bl_ii_ind_final[NWS_wgt.mask == True] = 0


    if test_plot:plt.plot(tmp_xypos_dict['LON'][NWS_bl_jj_ind_final,NWS_bl_ii_ind_final].ravel(), tmp_xypos_dict['LAT'][NWS_bl_jj_ind_final,NWS_bl_ii_ind_final].ravel(),'.') 

    '''
    NWS_nn_ii_ind = np.ma.minimum(np.ma.maximum(NWS_nn_ii_ind,0),ncurlon-1)
    NWS_nn_jj_ind = np.ma.minimum(np.ma.maximum(NWS_nn_jj_ind,0),ncurlat-1)

    NWS_bl_ii_ind_final = np.ma.minimum(np.ma.maximum(NWS_bl_ii_ind_final,0),ncurlon-1)
    NWS_bl_jj_ind_final = np.ma.minimum(np.ma.maximum(NWS_bl_jj_ind_final,0),ncurlat-1)
    '''

    #NWS_nn_ii_ind-=tmp_thd_cutx0
    #NWS_bl_ii_ind_final-=tmp_thd_cutx0
    #NWS_nn_jj_ind-=tmp_thd_cuty0
    #NWS_bl_jj_ind_final-=tmp_thd_cuty0
    
    NWS_nn_ii_ind = np.ma.minimum(np.ma.maximum(NWS_nn_ii_ind,0),nxylon-1)
    NWS_nn_jj_ind = np.ma.minimum(np.ma.maximum(NWS_nn_jj_ind,0),nxylat-1)

    NWS_bl_ii_ind = np.ma.minimum(np.ma.maximum(NWS_bl_ii_ind_final,0),nxylon-1)
    NWS_bl_jj_ind = np.ma.minimum(np.ma.maximum(NWS_bl_jj_ind_final,0),nxylat-1)


    #   ncurlat, ncurlon

    #pdb.set_trace()

    '''

    NWS_bl_ii_ind_out = np.floor((NWS_bl_ii_ind_final - tmp_thd_x0)).astype('int')  #/tmp_thd_dx).astype('int')
    NWS_bl_jj_ind_out = np.floor((NWS_bl_jj_ind_final - tmp_thd_y0)).astype('int')  #/tmp_thd_dy).astype('int')
    NWS_nn_ii_ind_out = np.floor((NWS_nn_ii_ind - tmp_thd_x0)).astype('int')  #/tmp_thd_dx).astype('int')
    NWS_nn_jj_ind_out = np.floor((NWS_nn_jj_ind - tmp_thd_y0)).astype('int')  #/tmp_thd_dy).astype('int')

    '''
    NWS_bl_jj_ind_out,NWS_bl_ii_ind_out = NWS_bl_jj_ind,NWS_bl_ii_ind
    NWS_nn_jj_ind_out,NWS_nn_ii_ind_out = NWS_nn_jj_ind,NWS_nn_ii_ind

    #return NWS_bl_jj_ind_out, NWS_bl_ii_ind_out, NWS_wgt, NWS_nn_jj_ind_out, NWS_nn_ii_ind_out
    
    if test_plot:plt.plot(tmp_xypos_dict['LON'][NWS_bl_jj_ind,NWS_bl_ii_ind].ravel(), tmp_xypos_dict['LAT'][NWS_bl_jj_ind,NWS_bl_ii_ind].ravel(),'.') 
    if test_plot:plt.plot(tmp_xypos_dict['LON'][NWS_nn_jj_ind,NWS_nn_ii_ind].ravel(), tmp_xypos_dict['LAT'][NWS_nn_jj_ind,NWS_nn_ii_ind].ravel(),'.') 

    sel_ii_out = np.floor((tmp_xypos_dict['XPOS'][NWS_nn_jj_ind_out,NWS_nn_ii_ind_out] - tmp_thd_x0)/tmp_thd_dx).astype('int')
    sel_jj_out = np.floor((tmp_xypos_dict['YPOS'][NWS_nn_jj_ind_out,NWS_nn_ii_ind_out] - tmp_thd_y0)/tmp_thd_dy).astype('int')
    sel_ii_out-=tmp_thd_cutx0
    sel_jj_out-=tmp_thd_cuty0
    #
    sel_ii_out = np.ma.minimum(np.ma.maximum(sel_ii_out,0),ncurlon-1)
    sel_jj_out = np.ma.minimum(np.ma.maximum(sel_jj_out,0),ncurlat-1)

    sel_bl_ii_out = np.floor((tmp_xypos_dict['XPOS'][NWS_bl_jj_ind_out,NWS_bl_ii_ind_out] - tmp_thd_x0)/tmp_thd_dx).astype('int')
    sel_bl_jj_out = np.floor((tmp_xypos_dict['YPOS'][NWS_bl_jj_ind_out,NWS_bl_ii_ind_out] - tmp_thd_y0)/tmp_thd_dy).astype('int')
    sel_bl_ii_out-=tmp_thd_cutx0
    sel_bl_jj_out-=tmp_thd_cuty0
    #
    sel_bl_ii_out = np.ma.minimum(np.ma.maximum(sel_bl_ii_out,0),ncurlon-1)
    sel_bl_jj_out = np.ma.minimum(np.ma.maximum(sel_bl_jj_out,0),ncurlat-1)


    if test_plot:plt.plot(tmp_lon_d_2[sel_jj_out,sel_ii_out].ravel(), tmp_lat_d_2[sel_jj_out,sel_ii_out].ravel(),'+')
    if test_plot:plt.plot(tmp_lon_d_2[sel_bl_jj_out,sel_bl_ii_out].ravel(), tmp_lat_d_2[sel_bl_jj_out,sel_bl_ii_out].ravel(),'x')


    ##   check that all points are inside available data - otherwise skip
    edge_ii = np.concatenate((tmp_lon_d_2[0,:], tmp_lon_d_2[:,-1], tmp_lon_d_2[-1,::-1], tmp_lon_d_2[::-1,0]))
    edge_jj = np.concatenate((tmp_lat_d_2[0,:], tmp_lat_d_2[:,-1], tmp_lat_d_2[-1,::-1], tmp_lat_d_2[::-1,0]))
    polygon = [[edii,edjj] for edii,edjj in zip(edge_ii,edge_jj)]
    points = np.array(([loni.ravel(),latj.ravel()])).T

    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

    import matplotlib.path as mpltPath

    path = mpltPath.Path(polygon)
    inside2 = path.contains_points(points)
    
    mask_out_of_area =  inside2.reshape(latj.shape) == False
    
    NWS_wgt.mask = NWS_wgt.mask | np.repeat(mask_out_of_area[np.newaxis],4,axis = 0)

    #pdb.set_trace()




    #sel_ii_out-=tmp_thd_cutx0
    #sel_jj_out-=tmp_thd_cuty0
    #sel_bl_ii_out-=tmp_thd_cutx0
    #sel_bl_jj_out-=tmp_thd_cuty0

    return sel_bl_jj_out, sel_bl_ii_out, NWS_wgt, sel_jj_out, sel_ii_out




def ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,loni,latj, XYPOS_ind_extended_NN = True,meth = 'bilin', verbose = False):
    '''
    For a given lon and lat (loni, latj), find the ii,jj index of a given dataset.

    If the xypos files are available, use them, otherwise, the method depends on the config.                   
    
    
    '''
    XYPOS_ind_extended_NN = True


    #meth = 'nearest'


    #Numeric code of the dataset
    th_d_ind = int(tmp_datstr[8:])

    # if using xypos files:
    if xypos_dict[tmp_datstr]['do_xypos'] == True:
        
        #Find size of xypos and config lat arrays. 
        nxylat, nxylon = xypos_dict[tmp_datstr]['LAT'].shape
        ndatlat, ndatlon = lat_d[th_d_ind].shape

        # find nearest grid box in the xypos arrays using a y = mx + c method
        xy_i_ind = ((loni-xypos_dict[tmp_datstr]['lon_min'])/xypos_dict[tmp_datstr]['dlon']).astype('int')
        xy_j_ind = ((latj-xypos_dict[tmp_datstr]['lat_min'])/xypos_dict[tmp_datstr]['dlat']).astype('int')

        #ensure the indices are in the array
        xy_i_ind = np.ma.minimum(np.ma.maximum(xy_i_ind,0),nxylon-1)
        xy_j_ind = np.ma.minimum(np.ma.maximum(xy_j_ind,0),nxylat-1)


        # Convert indices to lon and lats with the XYPOS file
        #
        if meth == 'nearest':
            if XYPOS_ind_extended_NN:
                #Use XYPOS ind array extended with a nearest neighbour interpolation - so no masked values
                sel_ii_out = np.floor((xypos_dict[tmp_datstr]['XPOS_NN'][xy_j_ind,xy_i_ind] - thd[th_d_ind]['x0'])/thd[th_d_ind]['dx']).astype('int')
                sel_jj_out = np.floor((xypos_dict[tmp_datstr]['YPOS_NN'][xy_j_ind,xy_i_ind] - thd[th_d_ind]['y0'])/thd[th_d_ind]['dy']).astype('int')
            else:
                sel_ii_out = np.floor((xypos_dict[tmp_datstr]['XPOS'][xy_j_ind,xy_i_ind] - thd[th_d_ind]['x0'])/thd[th_d_ind]['dx']).astype('int')
                sel_jj_out = np.floor((xypos_dict[tmp_datstr]['YPOS'][xy_j_ind,xy_i_ind] - thd[th_d_ind]['y0'])/thd[th_d_ind]['dy']).astype('int')
            
        
        elif meth == 'bilin':


            # xypos i,j flt indexes for a given lon/lat
            xy_i_ind_flt = ((loni-xypos_dict[tmp_datstr]['lon_min'])/xypos_dict[tmp_datstr]['dlon'])
            xy_j_ind_flt = ((latj-xypos_dict[tmp_datstr]['lat_min'])/xypos_dict[tmp_datstr]['dlat'])
            if verbose: print(xy_i_ind_flt,xy_j_ind_flt)

            # previous and subsequent xypos indices with ceil and floor
            xy_i_ind_flt_0 = int(np.floor(xy_i_ind_flt))
            xy_i_ind_flt_1 = int(np.ceil(xy_i_ind_flt))
            xy_j_ind_flt_0 = int(np.floor(xy_j_ind_flt))
            xy_j_ind_flt_1 = int(np.ceil(xy_j_ind_flt))
            if verbose: print(xy_i_ind_flt_0,xy_i_ind_flt_1,xy_j_ind_flt_0,xy_j_ind_flt_1)

            # ensure the indices are within the xypos matrix
            xy_i_ind_flt_0 = np.ma.minimum(np.ma.maximum(xy_i_ind_flt_0,0),nxylon-1)
            xy_j_ind_flt_0 = np.ma.minimum(np.ma.maximum(xy_j_ind_flt_0,0),nxylat-1)
            xy_i_ind_flt_1 = np.ma.minimum(np.ma.maximum(xy_i_ind_flt_1,0),nxylon-1)
            xy_j_ind_flt_1 = np.ma.minimum(np.ma.maximum(xy_j_ind_flt_1,0),nxylat-1)
            if verbose: print(xy_i_ind_flt_0,xy_i_ind_flt_1,xy_j_ind_flt_0,xy_j_ind_flt_1)

            # distance from flt ind and edges 
            xy_i_ind_flt_d = xy_i_ind_flt_1 - xy_i_ind_flt_0
            xy_j_ind_flt_d = xy_j_ind_flt_1 - xy_j_ind_flt_0
            if verbose: print(xy_i_ind_flt_d,xy_j_ind_flt_d)

            # Distance from flt index to grid box edges (horiz and vert)
            xy_lrbt_dist_0 = (xy_i_ind_flt - xy_i_ind_flt_0)/(xy_i_ind_flt_d) # from left
            xy_lrbt_dist_1 = (xy_i_ind_flt_1 - xy_i_ind_flt)/(xy_i_ind_flt_d) # from right
            xy_lrbt_dist_2 = (xy_j_ind_flt - xy_j_ind_flt_0)/(xy_j_ind_flt_d) # from bottom
            xy_lrbt_dist_3 = (xy_j_ind_flt_1 - xy_j_ind_flt)/(xy_j_ind_flt_d) # from top
            if verbose: print(xy_lrbt_dist_0,xy_lrbt_dist_1,xy_lrbt_dist_2,xy_lrbt_dist_3,xy_lrbt_dist_0+xy_lrbt_dist_1,xy_lrbt_dist_2+xy_lrbt_dist_3 )

            # if the flt ind is exactly an int, floor and ceil is the same, and so the xy_i_ind_flt_d == 0
            #       if so, set weighting to 0.5 (otherwise nan and masked)
            if xy_i_ind_flt_d == 0:
                xy_lrbt_dist_0 = 0.5
                xy_lrbt_dist_1 = 0.5
            if xy_j_ind_flt_d == 0:
                xy_lrbt_dist_2 = 0.5
                xy_lrbt_dist_3 = 0.5

            if verbose: print(xy_lrbt_dist_0,xy_lrbt_dist_1,xy_lrbt_dist_2,xy_lrbt_dist_3,xy_lrbt_dist_0+xy_lrbt_dist_1,xy_lrbt_dist_2+xy_lrbt_dist_3)


            # Weightings for each of the edges.
            xy_wgt_0 = xy_lrbt_dist_1*xy_lrbt_dist_3 # BL: dist to TR
            xy_wgt_1 = xy_lrbt_dist_0*xy_lrbt_dist_3 # BR: dist to TL
            xy_wgt_2 = xy_lrbt_dist_1*xy_lrbt_dist_2 # TL: dist to BR
            xy_wgt_3 = xy_lrbt_dist_0*xy_lrbt_dist_2 # TR: dist to BR
            if verbose: print(xy_wgt_0,xy_wgt_1,xy_wgt_2,xy_wgt_3,xy_wgt_0+xy_wgt_1+xy_wgt_2+xy_wgt_3 )

            #if (xy_wgt_0 + xy_wgt_1 + xy_wgt_2 + xy_wgt_3) != 1:
            if np.isclose(xy_wgt_0 + xy_wgt_1 + xy_wgt_2 + xy_wgt_3, 1) == False:
                print('XYPOS Weigthing not adding to 1', (xy_wgt_0 + xy_wgt_1 + xy_wgt_2 + xy_wgt_3))
                pdb.set_trace()

            # (BL lon*BL wgt) + (BR lon*BR wgt) + (TL lon*TL wgt) + (TR lon*TR wgt)
            if XYPOS_ind_extended_NN:
                #Use XYPOS ind array extended with a nearest neighbour interpolation - so no masked values
                sel_ii_out_flt = xypos_dict[tmp_datstr]['XPOS_NN'][xy_j_ind_flt_0,xy_i_ind_flt_0]*xy_wgt_0 + xypos_dict[tmp_datstr]['XPOS_NN'][xy_j_ind_flt_0,xy_i_ind_flt_1]*xy_wgt_1 + xypos_dict[tmp_datstr]['XPOS_NN'][xy_j_ind_flt_1,xy_i_ind_flt_0]*xy_wgt_2 + xypos_dict[tmp_datstr]['XPOS_NN'][xy_j_ind_flt_1,xy_i_ind_flt_1]*xy_wgt_3
                sel_jj_out_flt = xypos_dict[tmp_datstr]['YPOS_NN'][xy_j_ind_flt_0,xy_i_ind_flt_0]*xy_wgt_0 + xypos_dict[tmp_datstr]['YPOS_NN'][xy_j_ind_flt_0,xy_i_ind_flt_1]*xy_wgt_1 + xypos_dict[tmp_datstr]['YPOS_NN'][xy_j_ind_flt_1,xy_i_ind_flt_0]*xy_wgt_2 + xypos_dict[tmp_datstr]['YPOS_NN'][xy_j_ind_flt_1,xy_i_ind_flt_1]*xy_wgt_3
            else:
                sel_ii_out_flt = xypos_dict[tmp_datstr]['XPOS'][xy_j_ind_flt_0,xy_i_ind_flt_0]*xy_wgt_0 + xypos_dict[tmp_datstr]['XPOS'][xy_j_ind_flt_0,xy_i_ind_flt_1]*xy_wgt_1 + xypos_dict[tmp_datstr]['XPOS'][xy_j_ind_flt_1,xy_i_ind_flt_0]*xy_wgt_2 + xypos_dict[tmp_datstr]['XPOS'][xy_j_ind_flt_1,xy_i_ind_flt_1]*xy_wgt_3
                sel_jj_out_flt = xypos_dict[tmp_datstr]['YPOS'][xy_j_ind_flt_0,xy_i_ind_flt_0]*xy_wgt_0 + xypos_dict[tmp_datstr]['YPOS'][xy_j_ind_flt_0,xy_i_ind_flt_1]*xy_wgt_1 + xypos_dict[tmp_datstr]['YPOS'][xy_j_ind_flt_1,xy_i_ind_flt_0]*xy_wgt_2 + xypos_dict[tmp_datstr]['YPOS'][xy_j_ind_flt_1,xy_i_ind_flt_1]*xy_wgt_3
           


            sel_ii_out = np.floor((sel_ii_out_flt - thd[th_d_ind]['x0'])/thd[th_d_ind]['dx']).astype('int')
            sel_jj_out = np.floor((sel_jj_out_flt - thd[th_d_ind]['y0'])/thd[th_d_ind]['dy']).astype('int')



            if verbose: print(sel_ii_out_flt,sel_jj_out_flt)
            # round and set to ind.
            #sel_ii_out = np.round(sel_ii_out_flt).astype('int')
            #sel_jj_out = np.round(sel_jj_out_flt).astype('int')
            if verbose: print(sel_ii_out,sel_jj_out)


        sel_ii_out-=thd[th_d_ind]['cutx0']
        sel_jj_out-=thd[th_d_ind]['cuty0']


        #Offset added as selected grid box is adjacent the clicked grid box
        sel_ii_out-=1
        sel_jj_out-=1
        
        #ensure the indices are in the array
        sel_ii_out = np.ma.minimum(np.ma.maximum(sel_ii_out,0),ndatlon-1)
        sel_jj_out = np.ma.minimum(np.ma.maximum(sel_jj_out,0),ndatlat-1)


        loni,latj,lon_d[1][sel_jj_out,sel_ii_out],lat_d[1][sel_jj_out,sel_ii_out]

    else:
        

        if configd[th_d_ind].upper() in ['AMM7','GULF18']:
            lon = lon_d[th_d_ind][0,:]
            lat = lat_d[th_d_ind][:,0]
                
            sel_ii_out = (np.abs(lon[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']] - loni)).argmin()
            sel_jj_out = (np.abs(lat[thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']] - latj)).argmin()

        elif configd[th_d_ind].upper() in ['AMM15','CO9P2']:
            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
            #sel_ii_out = np.minimum(np.maximum( np.round((lon_mat_rot - lon_rotamm15[thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].min())/(dlon_rotamm15*thd[th_d_ind]['dx'])).astype('int') ,0),nlon_rotamm15//thd[th_d_ind]['dx']-1)
            #sel_jj_out = np.minimum(np.maximum( np.round((lat_mat_rot - lat_rotamm15[thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']].min())/(dlat_rotamm15*thd[th_d_ind]['dx'])).astype('int') ,0),nlat_rotamm15//thd[th_d_ind]['dx']-1)
            sel_ii_out = np.minimum(np.maximum( np.round((lon_mat_rot - rot_dict[configd[1]]['lon_rot'][thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].min())/(rot_dict[configd[1]]['dlon']*thd[th_d_ind]['dx'])).astype('int') ,0),rot_dict[configd[1]]['nlon']//thd[th_d_ind]['dx']-1)
            sel_jj_out = np.minimum(np.maximum( np.round((lat_mat_rot - rot_dict[configd[1]]['lat_rot'][thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy']].min())/(rot_dict[configd[1]]['dlat']*thd[th_d_ind]['dx'])).astype('int') ,0),rot_dict[configd[1]]['nlat']//thd[th_d_ind]['dx']-1)
        elif configd[th_d_ind].upper() in ['ORCA025','ORCA025EXT','ORCA12','ORCA025ICE','ORCA12ICE']:
            sel_dist_mat = np.sqrt((lon_d[th_d_ind][:,:] - loni)**2 + (lat_d[th_d_ind][:,:] - latj)**2 )
            sel_jj_out,sel_ii_out = sel_dist_mat.argmin()//sel_dist_mat.shape[th_d_ind], sel_dist_mat.argmin()%sel_dist_mat.shape[th_d_ind]

        else:
            print('config not supported:', configd[th_d_ind], 'Using brute force')
            sel_dist_mat = np.sqrt((lon_d[th_d_ind][:,:] - loni)**2 + (lat_d[th_d_ind][:,:] - latj)**2 )
            sel_jj_out,sel_ii_out = sel_dist_mat.argmin()//sel_dist_mat.shape[th_d_ind], sel_dist_mat.argmin()%sel_dist_mat.shape[th_d_ind]

            #pdb.set_trace()

    if np.ma.is_masked((sel_jj_out*sel_ii_out).any()):
        print('XYPOS indices are masked')
        pdb.set_trace()

    return sel_jj_out,sel_ii_out




def lonlat_iijj_amm15(loni_in,latj_in):
    
    loni,latj  =np.ma.array(loni_in),np.ma.array(latj_in)
    from rotated_pole_grid import rotated_grid_from_amm15,reduce_rotamm15_grid #rotated_grid_to_amm15, 
    lon_rotamm15,lat_rotamm15 = reduce_rotamm15_grid()

    dlon_rotamm15 = (np.diff(lon_rotamm15)).mean()
    dlat_rotamm15 = (np.diff(lat_rotamm15)).mean()
    nlon_rotamm15 = lon_rotamm15.size
    nlat_rotamm15 = lat_rotamm15.size

    #do this for each set of loni and lati you need to convert... 
    # these can be arrays
    lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
    ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15.min())/dlon_rotamm15).astype('int'),0),nlon_rotamm15-1)
    jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15.min())/dlat_rotamm15).astype('int'),0),nlat_rotamm15-1)

    return ii,jj



if __name__ == "__main__":
    main()
