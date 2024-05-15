
import pdb,sys,os,cftime,socket

from datetime import datetime, timedelta

from netCDF4 import Dataset,num2date

import numpy as np



import matplotlib.pyplot as plt

#from python3_plotting_function import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region


from matplotlib.colors import LinearSegmentedColormap, ListedColormap


computername = socket.gethostname()
comp = 'linux'
if computername in ['xcel00','xcfl00']: comp = 'hpc'


"""
https://gmd.copernicus.org/articles/16/2515/2023/



ter depth (Saulter et al., 2017; Valiente
et al., 2021b). The grid resolution is of 3 km for water
depths larger than 40 m and 1.5 km for coastal cells with
water depths of less than 40 m (Fig. 2). The SMC grid is
based on a rotated North Pole at 37.5◦ N, 177.5◦ E, achieving
an evenly spaced mesh around the UK


"""
#from math import *


RotNPole_lon = 177.5
RotNPole_lat = 37.5
RotSPole_lon = RotNPole_lon-180
RotSPole_lat = RotNPole_lat*-1
NP_coor = np.array([RotNPole_lon,RotNPole_lat])
SP_coor = np.array([RotSPole_lon,RotSPole_lat])



def load_nc_dims(tmp_data):
    x_dim = 'x'
    y_dim = 'y'
    z_dim = 'deptht'
    t_dim = 'time_counter'
    #pdb.set_trace()
    
    nc_dims = [ss for ss in tmp_data._dims.keys()]

    poss_zdims = ['deptht','depthu','depthv','z']
    poss_tdims = ['time_counter','time','t']
    poss_xdims = ['x','X','lon']
    poss_ydims = ['y','Y','lat']
    #pdb.set_trace()
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
            #return



def set_perc_clim_pcolor_in_region(perc_in_min,perc_in_max, illtype = 'pcolor', perc = True, set_not_get = True,ax = None,sym = False):

    if ax is None:
        ax = plt.gca()
        plt.sca(ax)
    else:
        plt.sca(ax)

    '''
    if a pcolor/pcolormesh, use
        illtype = pcolor/pcolor mesh (default)
    if scatter, use
        illtype = scatter
    if what to specify absolute values, use perc = False
    if what to get clims, rather than set them, set_not_get = True.
    '''


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



    '''

    plt.pcolormesh(lon,lat,notide_SSS_seas)
    print notide_SSS_seas.shape
    plt.xlim(0,5)
    plt.ylim(50,55)

    for child in plt.gca().get_children():child.__class__.__name__

    tmp_data_mat = plt.gca().get_children()[0].get_array()
    for ss in dir(plt.gca().get_children()[0]): ss

    '''


def get_colorbar_values(cb, verbose = False):
    '''
    return cb.ax.get_yticks()
    '''


    #print ('Think this is simpler with Python3')
    return cb.ax.get_yticks()
    # cb = plt.colorbar()

    #cbtickes = [float(ss.get_text()) for ss in cb.ax.get_yticklabels()]

    #because it didn't like minus number (the minus was actually u'\u2212', replace it with a -
    ticks_strings = cb.ax.get_yticklabels()


    # there was a unicode issue. Sometimes matplotlib used u'\u2212' for a minus symbol, which is beyond the first 128 values.
    # therefore when it was read from the figure, it crashed.
    # Now, I check the first character to see if it has a value of 8722 (   ord(u'\u2212')   ), and if it does, use a '-' instead



    cbtickes = []
    for ss in ticks_strings:
        #pdb.set_trace()
        ss_str = ss.get_text()
        if ss_str== '': continue
        #print (ss_str)
        #pdb.set_trace()
        if ord(ss_str[0]) == 8722:
            ss_str = '-' + ss_str[1:]
        cbtickes.append(ss_str)

    # cbtickes = [float(ss.get_text().decode("utf-8").replace(u'\u2212','-')) for ss in ticks_strings]
    if verbose: print(cbtickes)

    return cbtickes



def field_gradient_2d(tmpdat,e1t,e2t,dir_grad = False):


    nlat,nlon = e1t.shape

    xs = e1t.cumsum(axis = 1)
    ys = e2t.cumsum(axis = 0)

    dtmpdat_dx_c = (tmpdat[1:-1,2:] - tmpdat[1:-1,:-2])/(0.001*2*(xs[1:-1,2:] - xs[1:-1,:-2]))
    dtmpdat_dy_c = (tmpdat[2:,1:-1] - tmpdat[:-2,1:-1])/(0.001*2*(ys[2:,1:-1] - ys[:-2,1:-1]))

    dtmpdat_dkm = np.sqrt( (dtmpdat_dx_c)**2 + (dtmpdat_dy_c)**2   )


    dtmpdat_dkm_out = np.ma.zeros((nlat,nlon))
    dtmpdat_dx_c_out = np.ma.zeros((nlat,nlon))
    dtmpdat_dy_c_out = np.ma.zeros((nlat,nlon))
    dtmpdat_dkm_out[:] = np.ma.masked
    dtmpdat_dx_c_out[:] = np.ma.masked
    dtmpdat_dy_c_out[:] = np.ma.masked
    dtmpdat_dkm_out[1:-1,1:-1] = dtmpdat_dkm
    dtmpdat_dx_c_out[1:-1,1:-1] = dtmpdat_dx_c
    dtmpdat_dy_c_out[1:-1,1:-1] = dtmpdat_dy_c
    if dir_grad:
        return dtmpdat_dkm_out, dtmpdat_dx_c_out, dtmpdat_dy_c_out
    else:
        return dtmpdat_dkm_out




#from nemo_forcings_functions import interp1dmat_wgt, interp1dmat_create_weight
#from nemo_forcings_functions import  nearbed_index,extract_nb,mask_stats,load_nearbed_index


def interp1dmat_wgt(indata, wgt_tuple):
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

def interp1dmat_create_weight(gdept,z_lev):


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
    gdept_ma = gdept
    gdept_ma_min = gdept_ma.min(axis = 0)
    gdept_ma_max = gdept_ma.max(axis = 0)
    gdept_ma_ptp = gdept_ma.ptp(axis = 0)

    if verbose_debugging: print('x_mat, y_mat', datetime.now())

    xind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
    yind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
    #for zi in range(nz): zind_mat[zi,:,:] = zi
    for xi in range(nlon): xind_mat[xi,:] = xi
    for yi in range(nlat): yind_mat[:,yi] = yi

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

    '''
    plt.pcolormesh(wgt1*gdept_ma[ind1,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()
    plt.pcolormesh(wgt2*gdept_ma[ind2,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()
    plt.pcolormesh(wgt1*gdept_ma[ind1,xind_mat,yind_mat] + wgt2*gdept_ma[ind2,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()

    interpval = wgt1*gdept_ma[ind1,xind_mat,yind_mat] + wgt2*gdept_ma[ind2,xind_mat,yind_mat]
    plt.pcolormesh(interpval) ; plt.colorbar() ; plt.show()

    '''

    if z_lev == 0:
        ind1[:,:]= 1
        ind2[:,:]= 0
        wgt1[:,:]= 0.
        wgt2[:,:]= 1.
        wgt_mask = gdept_ma[0] == 0.1

    return ind1, ind2, wgt1, wgt2, xind_mat,yind_mat, wgt_mask

    pdb.set_trace()




def lon_lat_to_str(lon,lat,lonlatstr_format = '%.2f'):
    
    degree_sign= u'\N{DEGREE SIGN}'
    #pdb.set_trace()
    
    if lat>=0:
        latstr = (lonlatstr_format+'%sN')%(abs(lat),degree_sign)
    else:
        latstr = (lonlatstr_format+'%sS')%(abs(lat),degree_sign)
    
    if lon>=0:
        lonstr = (lonlatstr_format+'%sE')%(abs(lon),degree_sign)
    else:
        lonstr = (lonlatstr_format+'%sW')%(abs(lon),degree_sign)

    lat_lon_str = '%s %s'%(latstr, lonstr)

    return lat_lon_str,lonstr,latstr




def ismask(tmpvar):

    ismask_out = False
    if isinstance(tmpvar,np.ma.core.MaskedArray):
        ismask_out = True

    return ismask_out


def nearbed_int_use_index_val(tmp3dmasknbivar,nbzindint,nbiindint,nbijndint,tmask):
    
    tmp_nb_mat = np.ma.masked_invalid(tmp3dmasknbivar[nbzindint,nbiindint,nbijndint])
    
    return tmp_nb_mat



def nearbed_int_index_val(tmp3dmasknbivar):
    nbzindint,nbiindint,nbijndint,tmask = nearbed_int_index_func(tmp3dmasknbivar)

    tmp_nb_mat = nearbed_int_use_index_val(tmp3dmasknbivar,nbzindint,nbiindint,nbijndint,tmask)
    
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


def nearbed_index(filename, variable_4d,nemo_nb_i_filename = 'nemo_nb_i.nc'):


    rootgrp = Dataset(filename, 'r', format='NETCDF3_CLASSIC')#NETCDF3_CLASSIC
    tmp_var = rootgrp.variables[variable_4d][0,:,:,:]
    rootgrp.close()


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
    if ((nbind*1).sum(axis = 0).min() != (nz-1)) | ((nbind*1).sum(axis = 0).max() != (nz-1)) :
        print("ERROR, nbind has found more than one near bed boxes...")
        pdb.set_trace()


    #pdb.set_trace()

    rootgrp_out = Dataset(nemo_nb_i_filename, 'w', format='NETCDF3_CLASSIC')
    rootgrp_out.createDimension('x',ni)
    rootgrp_out.createDimension('y',nj)
    rootgrp_out.createDimension('z',nz)
    nb_i_out = rootgrp_out.createVariable('nb_i','i4',('z','y','x',),fill_value = -99)
    tmask_out = rootgrp_out.createVariable('t_mask','i4',('z','y','x',),fill_value = -99)
    nb_i_out[:,:] = nbind
    tmask_out[:,:] = tmask
    rootgrp_out.close()


    rootgrp_in = Dataset(nemo_nb_i_filename, 'r', format='NETCDF3_CLASSIC')
    nb_i_in = (rootgrp_in.variables['nb_i'][:,:,:] == 1)
    tmask = (rootgrp_in.variables['t_mask'][:,:,:] == 1)

    return nbind,tmask


def load_nearbed_index(nemo_nb_i_filename):


    rootgrp_in = Dataset(nemo_nb_i_filename, 'r', format='NETCDF3_CLASSIC')
    nbind = (rootgrp_in.variables['nb_i'][:,:,:] == 1)
    tmask = (rootgrp_in.variables['t_mask'][:,:,:] == 1)
    rootgrp_in.close()

    return nbind,tmask

def extract_nb(var_in,nbind):


    if ismask(var_in):
        tmpvar = var_in.copy()
    else:
        tmpvar = np.ma.masked_equal(var_in.copy(), 0)

    tmpvar.mask = tmpvar.mask | nbind

    nbvar = tmpvar.sum(axis = 0)

    return nbvar



def extract_ss(var_in,nbind):

    tmpvar = var_in.copy()
    ssvar = tmpvar[0,:,:]

    return ssvar


def extract_ss_nb_df(var_in,nbind,mask_in):

    ismask = False
    if isinstance(var_in,np.ma.core.MaskedArray):
        ismask = True
    if ismask:
         var_mask_in = var_in
    else:
         var_mask_in = np.ma.array(var_in, mask = mask_in)


    nbvar = extract_nb(var_mask_in,nbind)
    ssvar = extract_ss(var_mask_in,nbind)
    dfvar = ssvar - nbvar
    return ssvar,nbvar,dfvar

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
    dens_out =  (part1  +  part2)   * 1000. + SIGO
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

    return dens_out




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

    if calc_TS_comp:
        return pea,pea_T,pea_S
    else:
        return pea




def lon_lat_to_str(lon,lat,lonlatstr_format = '%.2f'):
    
    degree_sign= u'\N{DEGREE SIGN}'
    #pdb.set_trace()
    
    if lat>=0:
        latstr = (lonlatstr_format+'%sN')%(abs(lat),degree_sign)
    else:
        latstr = (lonlatstr_format+'%sS')%(abs(lat),degree_sign)
    
    if lon>=0:
        lonstr = (lonlatstr_format+'%sE')%(abs(lon),degree_sign)
    else:
        lonstr = (lonlatstr_format+'%sW')%(abs(lon),degree_sign)

    lat_lon_str = '%s %s'%(latstr, lonstr)

    return lat_lon_str,lonstr,latstr


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
    

#def regrid_2nd_thin_params(amm_conv_dict,thin_2nd,thin_x0_2nd,thin_y0_2nd,nlon_amm,nlat_amm, nlon_amm_2nd,nlat_amm_2nd,thin,thin_x0,thin_y0,thin_x1,thin_y1):
def regrid_2nd_thin_params(amm_conv_dict,nlon_amm,nlat_amm, nlon_amm_2nd,nlat_amm_2nd,thd):

    thin_2nd,thin_x0_2nd,thin_y0_2nd =   thd[2]['dx'],thd[2]['x0'],thd[2]['y0']
    thin,thin_x0,thin_y0,thin_x1,thin_y1 = thd[1]['dx'],thd[1]['x0'],thd[1]['y0'],thd[1]['x1'],thd[1]['y1']

    ##Nearest neighbour thinning
    NWS_amm_nn_jj_ind_final = (amm_conv_dict['NWS_amm_nn_jj_ind'] - thin_y0_2nd) //thin_2nd
    NWS_amm_nn_ii_ind_final = (amm_conv_dict['NWS_amm_nn_ii_ind'] - thin_y0_2nd) //thin_2nd


    #tmp_arr_shape = amm_conv_dict['NWS_amm_flt_ii_ind'].shape #size of config
    #nlat_amm,nlon_amm = tmp_arr_shape
    NWS_amm_flt_ii_ind_thin_0 = (amm_conv_dict['NWS_amm_flt_ii_ind']- thin_x0_2nd) /thin_2nd 
    NWS_amm_flt_jj_ind_thin_0 = (amm_conv_dict['NWS_amm_flt_jj_ind']- thin_y0_2nd) /thin_2nd
    NWS_amm_bl_ii_ind_thin_0 = amm_conv_dict['NWS_amm_bl_ii_ind'].copy().astype('int')*0
    NWS_amm_bl_jj_ind_thin_0 = amm_conv_dict['NWS_amm_bl_jj_ind'].copy().astype('int')*0
    NWS_amm_bl_ii_ind_thin_0[0,:,:] = np.floor(NWS_amm_flt_ii_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_jj_ind_thin_0[0,:,:] = np.floor(NWS_amm_flt_jj_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_ii_ind_thin_0[1,:,:] = np.ceil( NWS_amm_flt_ii_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_jj_ind_thin_0[1,:,:] = np.floor(NWS_amm_flt_jj_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_ii_ind_thin_0[2,:,:] = np.floor(NWS_amm_flt_ii_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_jj_ind_thin_0[2,:,:] = np.ceil( NWS_amm_flt_jj_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_ii_ind_thin_0[3,:,:] = np.ceil( NWS_amm_flt_ii_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR
    NWS_amm_bl_jj_ind_thin_0[3,:,:] = np.ceil( NWS_amm_flt_jj_ind_thin_0[:,:]).astype('int') # BL, BR, TL, TR

    print()
    #create the distance to the thinned left, right, bottom and top, and normalised by the distance to the thinned boxes.
    #NWS_amm_lrbt_dist_thin_0 = np.zeros((4,) + tmp_arr_shape, dtype = 'float')
    NWS_amm_lrbt_dist_thin_0 = np.zeros((4,nlat_amm,nlon_amm), dtype = 'float')
    NWS_amm_lrbt_dist_thin_0[0,:,:] = (NWS_amm_flt_ii_ind_thin_0[:,:] - NWS_amm_bl_ii_ind_thin_0[0,:,:])/(NWS_amm_bl_ii_ind_thin_0[1,:,:] - NWS_amm_bl_ii_ind_thin_0[0,:,:]) # distance from left handside
    NWS_amm_lrbt_dist_thin_0[1,:,:] = (NWS_amm_bl_ii_ind_thin_0[1,:,:] - NWS_amm_flt_ii_ind_thin_0[:,:])/(NWS_amm_bl_ii_ind_thin_0[1,:,:] - NWS_amm_bl_ii_ind_thin_0[0,:,:]) # distance from right handside
    NWS_amm_lrbt_dist_thin_0[2,:,:] = (NWS_amm_flt_jj_ind_thin_0[:,:] - NWS_amm_bl_jj_ind_thin_0[0,:,:])/(NWS_amm_bl_jj_ind_thin_0[2,:,:] - NWS_amm_bl_jj_ind_thin_0[0,:,:]) # distance from bottom handside
    NWS_amm_lrbt_dist_thin_0[3,:,:] = (NWS_amm_bl_jj_ind_thin_0[2,:,:] - NWS_amm_flt_jj_ind_thin_0[:,:])/(NWS_amm_bl_jj_ind_thin_0[2,:,:] - NWS_amm_bl_jj_ind_thin_0[0,:,:]) # distance from top handside
    #create the weights for the thinned indices        
    #NWS_amm_wgt_post_thin_0 = np.ma.zeros((4,) + tmp_arr_shape, dtype = 'float')
    NWS_amm_wgt_post_thin_0 = np.ma.zeros((4,nlat_amm,nlon_amm), dtype = 'float')
    NWS_amm_wgt_post_thin_0[0,:,:] = (NWS_amm_lrbt_dist_thin_0[1,:,:]*NWS_amm_lrbt_dist_thin_0[3,:,:]) # BL: dist to TR
    NWS_amm_wgt_post_thin_0[1,:,:] = (NWS_amm_lrbt_dist_thin_0[0,:,:]*NWS_amm_lrbt_dist_thin_0[3,:,:]) # BR: dist to TL
    NWS_amm_wgt_post_thin_0[2,:,:] = (NWS_amm_lrbt_dist_thin_0[1,:,:]*NWS_amm_lrbt_dist_thin_0[2,:,:]) # TL: dist to BR
    NWS_amm_wgt_post_thin_0[3,:,:] = (NWS_amm_lrbt_dist_thin_0[0,:,:]*NWS_amm_lrbt_dist_thin_0[2,:,:]) # TR: dist to BR
    #Catch and wrapped ii index
    NWS_amm_bl_jj_ind_final,NWS_amm_bl_ii_ind_final = NWS_amm_bl_jj_ind_thin_0.copy(),NWS_amm_bl_ii_ind_thin_0.copy()
    NWS_amm_bl_ii_ind_final[NWS_amm_bl_ii_ind_final<0] = 0
    NWS_amm_bl_jj_ind_final[NWS_amm_bl_jj_ind_final<0] = 0
    NWS_amm_bl_ii_ind_final[NWS_amm_bl_ii_ind_final>=nlon_amm_2nd//thin_2nd] = nlon_amm_2nd//thin_2nd-1
    NWS_amm_bl_jj_ind_final[NWS_amm_bl_jj_ind_final>=nlat_amm_2nd//thin_2nd] = nlat_amm_2nd//thin_2nd-1
    # Mask weight values where the index is out of the domain
    NWS_amm_wgt_post_thin_0[:,(NWS_amm_bl_jj_ind_final<0).any(axis = 0)] = np.ma.masked
    NWS_amm_wgt_post_thin_0[:,(NWS_amm_bl_ii_ind_final<0).any(axis = 0)] = np.ma.masked
    NWS_amm_wgt_post_thin_0[:,(NWS_amm_bl_jj_ind_final>=nlat_amm_2nd//thin_2nd).any(axis = 0)] = np.ma.masked
    NWS_amm_wgt_post_thin_0[:,(NWS_amm_bl_ii_ind_final>=nlon_amm_2nd//thin_2nd).any(axis = 0)] = np.ma.masked

    #respect initial mask, so no extrapolation
    NWS_amm_wgt_post_thin_0.mask = NWS_amm_wgt_post_thin_0.mask|amm_conv_dict['NWS_amm_wgt'].mask
    
    # Mask indices where the weight is masked
    NWS_amm_bl_jj_ind_final[NWS_amm_wgt_post_thin_0.mask == True] = 0
    NWS_amm_bl_ii_ind_final[NWS_amm_wgt_post_thin_0.mask == True] = 0


    #Thin for output grid

    NWS_amm_bl_jj_ind_out = NWS_amm_bl_jj_ind_final[:,thin_y0:thin_y1:thin,thin_x0:thin_x1:thin]
    NWS_amm_bl_ii_ind_out = NWS_amm_bl_ii_ind_final[:,thin_y0:thin_y1:thin,thin_x0:thin_x1:thin]
    NWS_amm_wgt_out = NWS_amm_wgt_post_thin_0[:,thin_y0:thin_y1:thin,thin_x0:thin_x1:thin]
    NWS_amm_nn_jj_ind_out = NWS_amm_nn_jj_ind_final[thin_y0:thin_y1:thin,thin_x0:thin_x1:thin]
    NWS_amm_nn_ii_ind_out = NWS_amm_nn_ii_ind_final[thin_y0:thin_y1:thin,thin_x0:thin_x1:thin]


    #pdb.set_trace()
    
    return NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out

def regrid_iijj_ew_ns(tmp_lon,tmp_lat,tmp_lon_arr, tmp_lat_arr,ew_tmp_lon_arr,ew_tmp_lat_arr,ns_tmp_lon_arr,ns_tmp_lat_arr,thin_2nd,thin_y0_2nd,thin_y1_2nd,regrid_meth):


    # Convert jj,ii index from config grid to config_2nd grid
    ii_2nd_ind = (np.abs(tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd] - tmp_lon)).argmin()
    jj_2nd_ind = (np.abs(tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd] - tmp_lat)).argmin()

    
    # Find size and resolution of thinned config2 grid
    dlon_thin = (tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][1:] - tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][:-1]).mean()
    dlat_thin = (tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][1:] - tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][:-1]).mean()
    nlon_thin = tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd].size
    nlat_thin = tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd].size

    # convert ew slice from config to config_2nd grid
    ew_ii_2nd_ind = ((ew_tmp_lon_arr - tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])//dlon_thin).astype('int')
    ew_jj_2nd_ind = ((ew_tmp_lat_arr - tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])//dlat_thin).astype('int')



    # convert ew slice from config to config_2nd grid
    ns_ii_2nd_ind = ((ns_tmp_lon_arr - tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])//dlon_thin).astype('int')
    ns_jj_2nd_ind = ((ns_tmp_lat_arr - tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])//dlat_thin).astype('int')

    #Trim the output ew/nw slice to be the size of the domain.
    ew_ii_2nd_ind = np.minimum( np.maximum(ew_ii_2nd_ind,0),nlon_thin-1)
    ew_jj_2nd_ind = np.minimum( np.maximum(ew_jj_2nd_ind,0),nlat_thin-1)

    ns_ii_2nd_ind = np.minimum( np.maximum(ns_ii_2nd_ind,0),nlon_thin-1)
    ns_jj_2nd_ind = np.minimum( np.maximum(ns_jj_2nd_ind,0),nlat_thin-1)


    # If using bilinear regridding, calculate the ew and nw bilinear weighings. Don't run this if unneccessary
    if regrid_meth == 2:

        ## EW
        ########################
        # Floating point index
        ew_flt_ii_ind = ((ew_tmp_lon_arr - tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])/dlon_thin)
        ew_flt_jj_ind = ((ew_tmp_lat_arr - tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])/dlat_thin)

        # initialise arrays
        ew_lrbt_dist = np.zeros(((4,)+ew_flt_ii_ind.shape), dtype = 'float')
        ew_wgt = np.ma.zeros(((4,)+ew_flt_ii_ind.shape), dtype = 'float')
        ew_bl_ii_ind = np.zeros(((4,)+ew_flt_ii_ind.shape), dtype = 'int')
        ew_bl_jj_ind = np.zeros(((4,)+ew_flt_ii_ind.shape), dtype = 'int')

        # Convert floating point index into the corner grid boxes (bottom left, bottom right, top left, top right) 
        ew_bl_ii_ind[0,:] = np.floor(ew_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_jj_ind[0,:] = np.floor(ew_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_ii_ind[1,:] = np.ceil( ew_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_jj_ind[1,:] = np.floor(ew_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_ii_ind[2,:] = np.floor(ew_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_jj_ind[2,:] = np.ceil( ew_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_ii_ind[3,:] = np.ceil( ew_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ew_bl_jj_ind[3,:] = np.ceil( ew_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR

        # Find distance from the floating point index to the corners
        ew_lrbt_dist[0,:] = (ew_flt_ii_ind[:] - ew_bl_ii_ind[0,:])/(ew_bl_ii_ind[1,:] - ew_bl_ii_ind[0,:]) # distance from left handside
        ew_lrbt_dist[1,:] = (ew_bl_ii_ind[1,:] - ew_flt_ii_ind[:])/(ew_bl_ii_ind[1,:] - ew_bl_ii_ind[0,:]) # distance from right handside
        ew_lrbt_dist[2,:] = (ew_flt_jj_ind[:] - ew_bl_jj_ind[0,:])/(ew_bl_jj_ind[2,:] - ew_bl_jj_ind[0,:]) # distance from bottom handside
        ew_lrbt_dist[3,:] = (ew_bl_jj_ind[2,:] - ew_flt_jj_ind[:])/(ew_bl_jj_ind[2,:] - ew_bl_jj_ind[0,:]) # distance from top handside

        # Create the weights
        ew_wgt[0,:] = (ew_lrbt_dist[1,:]*ew_lrbt_dist[3,:]) # BL: dist to TR
        ew_wgt[1,:] = (ew_lrbt_dist[0,:]*ew_lrbt_dist[3,:]) # BR: dist to TL
        ew_wgt[2,:] = (ew_lrbt_dist[1,:]*ew_lrbt_dist[2,:]) # TL: dist to BR
        ew_wgt[3,:] = (ew_lrbt_dist[0,:]*ew_lrbt_dist[2,:]) # TR: dist to BR

        #mask weights for grid size.
        ew_bl_jj_ind_final,ew_bl_ii_ind_final = ew_bl_jj_ind.copy(),ew_bl_ii_ind.copy()
        ew_wgt[:,(ew_bl_jj_ind_final<0).any(axis = 0)] = np.ma.masked
        ew_wgt[:,(ew_bl_ii_ind_final<0).any(axis = 0)] = np.ma.masked
        ew_wgt[:,(ew_bl_jj_ind_final>=nlat_thin).any(axis = 0)] = np.ma.masked
        ew_wgt[:,(ew_bl_ii_ind_final>=nlon_thin).any(axis = 0)] = np.ma.masked

        ew_bl_jj_ind_final[ew_wgt.mask == True] = 0
        ew_bl_ii_ind_final[ew_wgt.mask == True] = 0

        ############################################################
        # NB need to still mask weights when given real data. 
        ############################################################



        ## NS
        ########################
        # Floating point index
        ns_flt_ii_ind = ((ns_tmp_lon_arr - tmp_lon_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])/dlon_thin)
        ns_flt_jj_ind = ((ns_tmp_lat_arr - tmp_lat_arr[thin_y0_2nd:thin_y1_2nd:thin_2nd][0])/dlat_thin)

        # initialise arrays
        ns_lrbt_dist = np.zeros(((4,)+ns_flt_ii_ind.shape), dtype = 'float')
        ns_wgt = np.ma.zeros(((4,)+ns_flt_ii_ind.shape), dtype = 'float')
        ns_bl_ii_ind = np.zeros(((4,)+ns_flt_ii_ind.shape), dtype = 'int')
        ns_bl_jj_ind = np.zeros(((4,)+ns_flt_ii_ind.shape), dtype = 'int')

        # Convert floating point index into the corner grid boxes (bottom left, bottom right, top left, top right) 
        ns_bl_ii_ind[0,:] = np.floor(ns_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_jj_ind[0,:] = np.floor(ns_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_ii_ind[1,:] = np.ceil( ns_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_jj_ind[1,:] = np.floor(ns_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_ii_ind[2,:] = np.floor(ns_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_jj_ind[2,:] = np.ceil( ns_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_ii_ind[3,:] = np.ceil( ns_flt_ii_ind[:]).astype('int') # BL, BR, TL, TR
        ns_bl_jj_ind[3,:] = np.ceil( ns_flt_jj_ind[:]).astype('int') # BL, BR, TL, TR

        # Find distance from the floating point index to the corners
        ns_lrbt_dist[0,:] = (ns_flt_ii_ind[:] - ns_bl_ii_ind[0,:])/(ns_bl_ii_ind[1,:] - ns_bl_ii_ind[0,:]) # distance from left handside
        ns_lrbt_dist[1,:] = (ns_bl_ii_ind[1,:] - ns_flt_ii_ind[:])/(ns_bl_ii_ind[1,:] - ns_bl_ii_ind[0,:]) # distance from right handside
        ns_lrbt_dist[2,:] = (ns_flt_jj_ind[:] - ns_bl_jj_ind[0,:])/(ns_bl_jj_ind[2,:] - ns_bl_jj_ind[0,:]) # distance from bottom handside
        ns_lrbt_dist[3,:] = (ns_bl_jj_ind[2,:] - ns_flt_jj_ind[:])/(ns_bl_jj_ind[2,:] - ns_bl_jj_ind[0,:]) # distance from top handside

        # Create the weights
        ns_wgt[0,:] = (ns_lrbt_dist[1,:]*ns_lrbt_dist[3,:]) # BL: dist to TR
        ns_wgt[1,:] = (ns_lrbt_dist[0,:]*ns_lrbt_dist[3,:]) # BR: dist to TL
        ns_wgt[2,:] = (ns_lrbt_dist[1,:]*ns_lrbt_dist[2,:]) # TL: dist to BR
        ns_wgt[3,:] = (ns_lrbt_dist[0,:]*ns_lrbt_dist[2,:]) # TR: dist to BR

        #mask weights for grid size.
        ns_bl_jj_ind_final,ns_bl_ii_ind_final = ns_bl_jj_ind.copy(),ns_bl_ii_ind.copy()
        ns_wgt[:,(ns_bl_jj_ind_final<0).any(axis = 0)] = np.ma.masked
        ns_wgt[:,(ns_bl_ii_ind_final<0).any(axis = 0)] = np.ma.masked
        ns_wgt[:,(ns_bl_jj_ind_final>=nlat_thin).any(axis = 0)] = np.ma.masked
        ns_wgt[:,(ns_bl_ii_ind_final>=nlon_thin).any(axis = 0)] = np.ma.masked

        ns_bl_jj_ind_final[ns_wgt.mask == True] = 0
        ns_bl_ii_ind_final[ns_wgt.mask == True] = 0

        ############################################################
        # NB need to still mask weights when given real data. 
        ############################################################


    else:
        ew_bl_ii_ind_final,ew_bl_jj_ind_final,ew_wgt, ns_bl_ii_ind_final,ns_bl_jj_ind_final,ns_wgt = 0,0,0,0,0,0



    return ii_2nd_ind,jj_2nd_ind,ew_ii_2nd_ind,ew_jj_2nd_ind,ns_ii_2nd_ind,ns_jj_2nd_ind, ew_bl_ii_ind_final,ew_bl_jj_ind_final,ew_wgt, ns_bl_ii_ind_final,ns_bl_jj_ind_final,ns_wgt

def vector_div(tmpU, tmpV, tmpdx, tmpdy):
    div_out = (np.gradient(tmpU, axis=0)/tmpdx) + (np.gradient(tmpV, axis=1)/tmpdy)

    return div_out



def vector_curl(tmpU, tmpV, tmpdx, tmpdy):
    curl_out = (np.gradient(tmpV, axis=0)/tmpdx) - (np.gradient(tmpU, axis=1)/tmpdy)

    return curl_out


def pycnocline_params(rho_4d,gdept_3d,e3t_3d):

    '''
    N2,Pync_Z,Pync_Th,N2_max = pycnocline_params(data_inst[tmp_datstr][np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])

    '''
    #pdb.set_trace()
    # vertical density gradient
    drho =  rho_4d[:,2:,:,:] -  rho_4d[:,:-2,:,:]
    dz = gdept_3d[2:,:,:] - gdept_3d[:-2,:,:]

    drho_dz = drho/dz
    
    # N, Brunt-Vaisala frequency
    # N**2
    N2 = rho_4d.copy()*0*np.ma.masked
    N2[:,1:-1,:,:]  = drho_dz*(-9.81/rho_4d[:,1:-1,:,:])
    N2[:,0,:,:]= N2[:,1,:,:]

    # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018JC014307
    # Equation 14

    
    # Pycnocline Depth:
    Pync_Z = ((N2*gdept_3d)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1)
                    
    # Pycnocline thickness:
    Pync_Th  = np.sqrt(((N2*(gdept_3d-Pync_Z)**2)*e3t_3d).sum(axis = 1)/(N2*e3t_3d).sum(axis = 1))


    # Depth of max Nz
    # find array size
    n_t,n_z, n_j, n_i = rho_4d.shape

    # Make dummy index array
    n_i_mat, n_j_mat = np.meshgrid(range(n_i), range(n_j))

    # find index of maximum N2 depth
    N2_max_arg = N2.argmax(axis = 1)

    # use gdept to calcuate these as a depth
    N2_max = gdept_3d[N2_max_arg,np.tile(n_j_mat,(n_t,1,1)),np.tile(n_i_mat,(n_t,1,1))]

    return N2,Pync_Z,Pync_Th,N2_max
                      

def reload_data_instances(var,thd,ldi,ti,var_grid, xarr_dict, grid_dict,var_dim,load_2nd_files):

    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

    data_inst = {}
    tmp_datstr_mat = [ss for ss in xarr_dict.keys()]

    start_time_load_inst = datetime.now()
    #pdb.set_trace()
    for tmp_datstr in tmp_datstr_mat:
        th_d_ind = int(tmp_datstr[-1])
        if var == 'N:P':
            
            map_dat_N_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['N3n'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            map_dat_P_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['N1p'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            data_inst[tmp_datstr] = map_dat_N_1/map_dat_P_1
            del(map_dat_N_1)
            del(map_dat_P_1)

        elif var in ['baroc_mag','baroc_div','baroc_curl']:
            
            map_dat_3d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_U][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            map_dat_3d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_V][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            if    var == 'baroc_mag': data_inst[tmp_datstr] = np.sqrt(map_dat_3d_U_1**2 + map_dat_3d_V_1**2)
            elif  var == 'baroc_div': data_inst[tmp_datstr] = vector_div(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict[tmp_datstr]['e1t']*thd[th_d_ind]['dx'],grid_dict[tmp_datstr]['e2t']*thd[th_d_ind]['dx'])
            elif var == 'baroc_curl': data_inst[tmp_datstr] = vector_curl(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict[tmp_datstr]['e1t']*thd[th_d_ind]['dx'],grid_dict[tmp_datstr]['e2t']*thd[th_d_ind]['dx'])
            del(map_dat_3d_U_1)
            del(map_dat_3d_V_1)


        elif var in ['barot_mag','barot_div','barot_curl']:
            tmp_var_Ubar = 'ubar'
            tmp_var_Vbar = 'vbar'
            
            map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['U'][ldi].variables[tmp_var_Ubar][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr]['V'][ldi].variables[tmp_var_Vbar][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            if    var == 'barot_mag': data_inst[tmp_datstr] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
            elif  var == 'barot_div': data_inst[tmp_datstr] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
            elif var == 'barot_curl': data_inst[tmp_datstr] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict[tmp_datstr]['e1t'],grid_dict[tmp_datstr]['e2t'])
            del(map_dat_2d_U_1)
            del(map_dat_2d_V_1)

         

        elif var.upper() in ['PEA', 'PEAT','PEAS']:

            gdept_mat = grid_dict[tmp_datstr]['gdept'][np.newaxis]
            dz_mat = grid_dict[tmp_datstr]['e3t'][np.newaxis]

            tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(tmp_T_data_1[np.newaxis],tmp_S_data_1[np.newaxis],gdept_mat,dz_mat,calc_TS_comp = True ) 
            if var.upper() == 'PEA':
                data_inst[tmp_datstr]= tmppea_1[0]
            elif var.upper() == 'PEAT':
                data_inst[tmp_datstr] = tmppeat_1[0]
            elif var.upper() == 'PEAS':
                data_inst[tmp_datstr] = tmppeas_1[0]


        elif var.upper() in ['RHO','N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
            #tmp_rho = {}
            tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
            tmp_rho = sw_dens(tmp_T_data_1,tmp_S_data_1) 
            

            if var.upper() =='RHO'.upper():
                data_inst[tmp_datstr]=tmp_rho

            elif var.upper() in ['N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper(),'N2max'.upper()]:
                
                N2,Pync_Z,Pync_Th,N2_max = pycnocline_params(tmp_rho[np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])
            
                if var.upper() =='N2'.upper():data_inst[tmp_datstr]=N2[0]
                elif var.upper() =='Pync_Z'.upper():data_inst[tmp_datstr]=Pync_Z[0]
                elif var.upper() =='Pync_Th'.upper():data_inst[tmp_datstr]=Pync_Th[0]
                elif var.upper() =='N2max'.upper():data_inst[tmp_datstr]=N2_max[0]
            '''
            tmp_rho = {}
            for tmp_datstr in tmp_datstr_mat: 
                tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                tmp_rho[tmp_datstr] = sw_dens(tmp_T_data_1,tmp_S_data_1) 
                

                if var.upper() =='RHO'.upper():
                    data_inst[tmp_datstr]=tmp_rho[tmp_datstr]

                elif var.upper() in ['N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper()]:
                        
                    for tmp_datstr in tmp_datstr_mat: 
                        N2,Pync_Z,Pync_Th,N2_max = pycnocline_params(tmp_rho[tmp_datstr][np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])
                    
                        if var.upper() =='N2'.upper():data_inst[tmp_datstr]=N2[0]
                        elif var.upper() =='Pync_Z'.upper():data_inst[tmp_datstr]=Pync_Z[0]
                        elif var.upper() =='Pync_Th'.upper():data_inst[tmp_datstr]=Pync_Th[0]

            '''

        else:
            if var_dim[var] == 3:
                data_inst[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
                
            if var_dim[var] == 4:
                data_inst[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']].load())
               

    preload_data_ti = ti
    preload_data_var = var
    preload_data_ldi = ldi
    print('======================================')
    print('Reloaded data instances for ti = %i, var = %s %s = %s'%(ti,var,datetime.now(),datetime.now() - start_time_load_inst))


    return data_inst,preload_data_ti,preload_data_var,preload_data_ldi


    
"""
def reload_data_instances(var,thd,ldi,ti,var_grid, xarr_dict, grid_dict,var_dim,load_2nd_files):

    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

    data_inst = {}

    if load_2nd_files:
        tmp_datstr_mat = ['Dataset 1','Dataset 2']
    else:    
        tmp_datstr_mat = ['Dataset 1']

    start_time_load_inst = datetime.now()
    if var == 'N:P':
        
        map_dat_N_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables['N3n'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        map_dat_P_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables['N1p'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        data_inst['Dataset 1'] = map_dat_N_1/map_dat_P_1
        del(map_dat_N_1)
        del(map_dat_P_1)

        if load_2nd_files:
            map_dat_N_1 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables['N3n'][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            map_dat_P_1 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables['N1p'][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            data_inst['Dataset 2'] = map_dat_N_1/map_dat_P_1
            del(map_dat_N_1)
            del(map_dat_P_1)
        else:
            data_inst['Dataset 2'] = data_inst['Dataset 1']
    elif var in ['baroc_mag','baroc_div','baroc_curl']:
        
        map_dat_3d_U_1 = np.ma.masked_invalid(xarr_dict['Dataset 1']['U'][ldi].variables[tmp_var_U][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        map_dat_3d_V_1 = np.ma.masked_invalid(xarr_dict['Dataset 1']['V'][ldi].variables[tmp_var_V][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        if    var == 'baroc_mag': data_inst['Dataset 1'] = np.sqrt(map_dat_3d_U_1**2 + map_dat_3d_V_1**2)
        elif  var == 'baroc_div': data_inst['Dataset 1'] = vector_div(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict['Dataset 1']['e1t']*thd[1]['dx'],grid_dict['Dataset 1']['e2t']*thd[1]['dx'])
        elif var == 'baroc_curl': data_inst['Dataset 1'] = vector_curl(map_dat_3d_U_1, map_dat_3d_V_1,grid_dict['Dataset 1']['e1t']*thd[1]['dx'],grid_dict['Dataset 1']['e2t']*thd[1]['dx'])
        del(map_dat_3d_U_1)
        del(map_dat_3d_V_1)

        if load_2nd_files:
            map_dat_3d_U_2 = np.ma.masked_invalid(xarr_dict['Dataset 2']['U'][ldi].variables[tmp_var_U][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            map_dat_3d_V_2 = np.ma.masked_invalid(xarr_dict['Dataset 2']['V'][ldi].variables[tmp_var_V][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            data_inst['Dataset 2'] = np.sqrt(map_dat_3d_U_2**2 + map_dat_3d_V_2**2)
            if    var == 'baroc_mag': data_inst['Dataset 2'] = np.sqrt(map_dat_3d_U_2**2 + map_dat_3d_V_2**2)
            elif  var == 'baroc_div': data_inst['Dataset 2'] = vector_div(map_dat_3d_U_2, map_dat_3d_V_2,grid_dict['Dataset 2']['e1t'],grid_dict['Dataset 2']['e2t']*thd[2]['dx'])
            elif var == 'baroc_curl': data_inst['Dataset 2'] = vector_curl(map_dat_3d_U_2, map_dat_3d_V_2,grid_dict['Dataset 2']['e1t'],grid_dict['Dataset 2']['e2t']*thd[2]['dx'])
            del(map_dat_3d_U_2)
            del(map_dat_3d_V_2)
        else:
            data_inst['Dataset 2'] = data_inst['Dataset 1']


    elif var in ['barot_mag','barot_div','barot_curl']:
        tmp_var_Ubar = 'ubar'
        tmp_var_Vbar = 'vbar'
        
        map_dat_2d_U_1 = np.ma.masked_invalid(xarr_dict['Dataset 1']['U'][ldi].variables[tmp_var_Ubar][ti,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        map_dat_2d_V_1 = np.ma.masked_invalid(xarr_dict['Dataset 1']['V'][ldi].variables[tmp_var_Vbar][ti,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        if    var == 'barot_mag': data_inst['Dataset 1'] = np.sqrt(map_dat_2d_U_1**2 + map_dat_2d_V_1**2)
        elif  var == 'barot_div': data_inst['Dataset 1'] = vector_div(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict['Dataset 1']['e1t'],grid_dict['Dataset 1']['e2t'])
        elif var == 'barot_curl': data_inst['Dataset 1'] = vector_curl(map_dat_2d_U_1, map_dat_2d_V_1,grid_dict['Dataset 1']['e1t'],grid_dict['Dataset 1']['e2t'])
        del(map_dat_2d_U_1)
        del(map_dat_2d_V_1)

        if load_2nd_files:
            map_dat_2d_U_2 = np.ma.masked_invalid(xarr_dict['Dataset 2']['U'][ldi].variables[tmp_var_Ubar][ti,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            map_dat_2d_V_2 = np.ma.masked_invalid(xarr_dict['Dataset 2']['V'][ldi].variables[tmp_var_Vbar][ti,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            if    var == 'baroc_mag': data_inst['Dataset 2'] = np.sqrt(map_dat_2d_U_2**2 + map_dat_2d_V_2**2)
            elif  var == 'baroc_div': data_inst['Dataset 2'] = vector_div(map_dat_2d_U_2, map_dat_2d_V_2,grid_dict['Dataset 2']['e1t'],grid_dict['Dataset 2']['e2t'])
            elif var == 'baroc_curl': data_inst['Dataset 2'] = vector_curl(map_dat_2d_U_2, map_dat_2d_V_2,grid_dict['Dataset 2']['e1t'],grid_dict['Dataset 2']['e2t'])
            del(map_dat_2d_U_2)
            del(map_dat_2d_V_2)
        else:
            data_inst['Dataset 2'] = data_inst['Dataset 1']


    elif var.upper() in ['PEA', 'PEAT','PEAS']:

        gdept_mat = grid_dict['Dataset 1']['gdept'][np.newaxis]
        dz_mat = grid_dict['Dataset 1']['e3t'][np.newaxis]

        tmp_T_data_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables['votemper'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        tmp_S_data_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
        tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(tmp_T_data_1[np.newaxis],tmp_S_data_1[np.newaxis],gdept_mat,dz_mat,calc_TS_comp = True ) 
        if var.upper() == 'PEA':
            data_inst['Dataset 1']= tmppea_1[0]
        elif var.upper() == 'PEAT':
            data_inst['Dataset 1'] = tmppeat_1[0]
        elif var.upper() == 'PEAS':
            data_inst['Dataset 1'] = tmppeas_1[0]

        
        if load_2nd_files:
            gdept_mat_2nd = grid_dict['Dataset 2']['gdept'][np.newaxis]
            dz_mat_2nd = grid_dict['Dataset 2']['e3t'][np.newaxis]
            tmp_T_data_2 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables['votemper'][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            tmp_S_data_2 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
            tmppea_2, tmppeat_2, tmppeas_2 = pea_TS(tmp_T_data_2[np.newaxis],tmp_S_data_2[np.newaxis],gdept_mat_2nd,dz_mat_2nd,calc_TS_comp = True ) 


            if var.upper() == 'PEA':
                data_inst['Dataset 2'] = tmppea_2[0]
            elif var.upper() == 'PEAT':
                data_inst['Dataset 2'] = tmppeat_2[0]
            elif var.upper() == 'PEAS':
                data_inst['Dataset 2'] = tmppeas_2[0]
        else:
            data_inst['Dataset 2'] = data_inst['Dataset 1']

    elif var.upper() in ['RHO','N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper()]:

        tmp_rho = {}
        for tmp_datstr in tmp_datstr_mat: 
            tmp_T_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['votemper'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
            tmp_S_data_1 = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables['vosaline'][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
            tmp_rho[tmp_datstr] = sw_dens(tmp_T_data_1,tmp_S_data_1) 
            

            if var.upper() =='RHO'.upper():
                data_inst[tmp_datstr]=tmp_rho[tmp_datstr]

            elif var.upper() in ['N2'.upper(),'Pync_Z'.upper(),'Pync_Th'.upper()]:
                    
                for tmp_datstr in tmp_datstr_mat: 
                    N2,Pync_Z,Pync_Th,N2_max = pycnocline_params(tmp_rho[tmp_datstr][np.newaxis],grid_dict[tmp_datstr]['gdept'],grid_dict[tmp_datstr]['e3t'])
                
                    if var.upper() =='N2'.upper():data_inst[tmp_datstr]=N2[0]
                    elif var.upper() =='Pync_Z'.upper():data_inst[tmp_datstr]=Pync_Z[0]
                    elif var.upper() =='Pync_Th'.upper():data_inst[tmp_datstr]=Pync_Th[0]

        if load_2nd_files == False:
            data_inst['Dataset 2'] = data_inst['Dataset 1']


    else:
        if var_dim[var] == 3:
                
            data_inst['Dataset 1'] = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][ti,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
            data_inst['Dataset 2'] = data_inst['Dataset 1']
            if load_2nd_files:
                data_inst['Dataset 2'] = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][ti,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())
                
        if var_dim[var] == 4:
            data_inst['Dataset 1'] = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][ti,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].load())
            data_inst['Dataset 2'] = data_inst['Dataset 1']
            if load_2nd_files:
                data_inst['Dataset 2'] = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][ti,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']].load())

    preload_data_ti = ti
    preload_data_var = var
    preload_data_ldi = ldi
    print('======================================')
    print('Reloaded data instances for ti = %i, var = %s %s = %s'%(ti,var,datetime.now(),datetime.now() - start_time_load_inst))
    return data_inst,preload_data_ti,preload_data_var,preload_data_ldi


"""

def reload_map_data_comb(var,ldi,ti,z_meth,zz,zi, data_inst,var_dim,interp1d_ZwgtT,grid_dict,nav_lon,nav_lat,regrid_params,regrid_meth,thd,configd,load_2nd_files):

    if var_dim[var] == 3:
        map_dat_dict = reload_map_data_comb_2d(var,ldi,ti, data_inst,grid_dict,regrid_params,regrid_meth ,thd,configd[2],load_2nd_files)

    else:
        if z_meth == 'z_slice':
            map_dat_dict = reload_map_data_comb_zmeth_zslice(zz, data_inst,interp1d_ZwgtT,grid_dict,regrid_params,regrid_meth ,thd,configd[2],load_2nd_files)
        elif z_meth in ['nb','df','zm']:
            map_dat_dict = reload_map_data_comb_zmeth_nb_df_zm_3d(z_meth, data_inst,grid_dict,regrid_params,regrid_meth ,thd,configd[2],load_2nd_files)
        elif z_meth in ['ss']:
            map_dat_dict = reload_map_data_comb_zmeth_ss_3d(data_inst,regrid_params,regrid_meth,thd,configd[2],load_2nd_files)
        elif z_meth == 'z_index':
            map_dat_dict = reload_map_data_comb_zmeth_zindex(data_inst,zi,regrid_params,regrid_meth,thd,configd[2],load_2nd_files)
        else:
            print('z_meth not supported:',z_meth)
            pdb.set_trace()

    map_dat_dict['x'] = nav_lon
    map_dat_dict['y'] = nav_lat

    return map_dat_dict
            


def reload_map_data_comb_2d(var,ldi,ti, data_inst,grid_dict,regrid_params,regrid_meth,thd,configd,load_2nd_files): # ,
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  

    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = data_inst['Dataset 1']
    for tmp_datstr in tmp_datstr_mat_secondary:
        map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params,regrid_meth,thd,configd,data_inst[tmp_datstr])

    '''
    map_dat_1 = data_inst['Dataset 1']
    map_dat_2 = map_dat_1
    
    if load_2nd_files:
        map_dat_2 = regrid_2nd(regrid_params,regrid_meth,thd,configd,data_inst['Dataset 2'])
        
    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = map_dat_1
    map_dat_dict['Dataset 2'] = map_dat_2
    '''
    
    return map_dat_dict



def  reload_map_data_comb_zmeth_ss_3d(data_inst,regrid_params,regrid_meth,thd,configd,load_2nd_files):
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  

    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = data_inst['Dataset 1'][0]
    for tmp_datstr in tmp_datstr_mat_secondary:
        map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params,regrid_meth,thd,configd,data_inst[tmp_datstr][0])

    '''

    # load files
    map_dat_ss_1 = data_inst['Dataset 1'][0]
    if load_2nd_files:
            map_dat_ss_2 = regrid_2nd(regrid_params,regrid_meth,thd,configd,data_inst['Dataset 2'][0])
    else: 
        map_dat_ss_2 = map_dat_ss_1

    map_dat_dict= {}
    map_dat_dict['Dataset 1'] = map_dat_1
    map_dat_dict['Dataset 2'] = map_dat_2

    '''

    return map_dat_dict

def reload_map_data_comb_zmeth_nb_df_zm_3d(z_meth, data_inst,grid_dict,regrid_params,regrid_meth,thd,configd,load_2nd_files):
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    # load files
    map_dat_3d_1 = data_inst['Dataset 1']
    

    # process onto 2d levels
    map_dat_ss_1 = map_dat_3d_1[0]
    map_dat_nb_1 = nearbed_int_index_val(map_dat_3d_1)
    map_dat_zm_1 = weighted_depth_mean_masked_var(map_dat_3d_1,grid_dict['Dataset 1']['e3t'])
    del(map_dat_3d_1)
    map_dat_df_1 = map_dat_ss_1 - map_dat_nb_1


    if z_meth == 'nb': map_dat_dict['Dataset 1'] = map_dat_nb_1
    if z_meth == 'df': map_dat_dict['Dataset 1'] = map_dat_ss_1 - map_dat_nb_1
    if z_meth == 'zm': map_dat_dict['Dataset 1'] = map_dat_zm_1

    for tmp_datstr in tmp_datstr_mat_secondary:
    
        map_dat_3d_2 = np.ma.masked_invalid(data_inst[tmp_datstr])

        map_dat_ss_2 = regrid_2nd(regrid_params,regrid_meth,thd,configd,map_dat_3d_2[0])
        map_dat_nb_2 = regrid_2nd(regrid_params,regrid_meth,thd,configd,nearbed_int_index_val(map_dat_3d_2))
        map_dat_zm_2 = regrid_2nd(regrid_params,regrid_meth,thd,configd,weighted_depth_mean_masked_var(map_dat_3d_2,grid_dict['Dataset 2']['e3t']))
        del(map_dat_3d_2)
        map_dat_df_2 = map_dat_ss_2 - map_dat_nb_2
        if z_meth == 'nb': map_dat_dict[tmp_datstr] = map_dat_nb_2
        if z_meth == 'df': map_dat_dict[tmp_datstr] = map_dat_ss_2 - map_dat_nb_2
        if z_meth == 'zm': map_dat_dict[tmp_datstr] = map_dat_zm_2

        
    return map_dat_dict


def reload_map_data_comb_zmeth_zslice(zz, data_inst,interp1d_ZwgtT,grid_dict,regrid_params,regrid_meth,thd,configd,load_2nd_files):
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    if zz not in interp1d_ZwgtT['Dataset 1'].keys():
        interp1d_ZwgtT['Dataset 1'][zz] = interp1dmat_create_weight(grid_dict['Dataset 1']['gdept'],zz)


    map_dat_3d_1 = np.ma.masked_invalid(data_inst['Dataset 1'])
    if load_2nd_files:
        map_dat_3d_2 = np.ma.masked_invalid(data_inst['Dataset 2'])
    else:
        map_dat_3d_2 = map_dat_3d_1

    map_dat_dict['Dataset 1'] =  interp1dmat_wgt(np.ma.masked_invalid(map_dat_3d_1),interp1d_ZwgtT['Dataset 1'][zz])

    
    for tmp_datstr in tmp_datstr_mat_secondary:
    
        if zz not in interp1d_ZwgtT[tmp_datstr].keys(): 
            interp1d_ZwgtT[tmp_datstr][zz] = interp1dmat_create_weight(grid_dict[tmp_datstr]['gdept'],zz)
        
        map_dat_dict[tmp_datstr] = regrid_2nd(regrid_params,regrid_meth,thd,configd,interp1dmat_wgt(np.ma.masked_invalid(map_dat_3d_2),interp1d_ZwgtT[tmp_datstr][zz]))
       

    return map_dat_dict


def reload_map_data_comb_zmeth_zindex(data_inst,zi,regrid_params,regrid_meth,thd,configd,load_2nd_files):
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  

    map_dat_dict= {}


    map_dat_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][zi])
    for tmp_datstr in tmp_datstr_mat_secondary:
    
        map_dat_dict[tmp_datstr]  = np.ma.masked_invalid(regrid_2nd(regrid_params,regrid_meth,thd,configd,data_inst[tmp_datstr][zi]))


    return map_dat_dict




def reload_ew_data_comb(ii,jj,ti,thd, data_inst, nav_lon, nav_lat, grid_dict,regrid_meth, iijj_ind,load_2nd_files,configd):
    tmp_datstr_mat = [ss for ss in data_inst.keys()]
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  
    '''
    reload the data for the E-W cross-section


    '''

    ew_slice_dict = {}



    ew_slice_dict['x'] =  nav_lon[jj,:]
    ew_slice_dict['y'] =  grid_dict['Dataset 1']['gdept'][:,jj,:]
    ew_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][:,jj,:])

    for tmp_datstr in tmp_datstr_mat_secondary:


        ew_ii_2nd_ind,ew_jj_2nd_ind,ew_bl_jj_ind_final,ew_bl_ii_ind_final,ew_wgt = iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_wgt']


        if configd[2] is None:
            ew_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,jj,:])
        else:
            if regrid_meth == 1:
                tmpdat_ew_slice = np.ma.masked_invalid(data_inst[tmp_datstr][:,ew_jj_2nd_ind,ew_ii_2nd_ind].T)
            elif regrid_meth == 2:            
                tmp_data_inst_2_bl = data_inst[tmp_datstr][:,ew_bl_jj_ind_final,ew_bl_ii_ind_final]
                tmp_ew_wgt = ew_wgt.copy()
                tmp_ew_wgt.mask = tmp_ew_wgt.mask | tmp_data_inst_2_bl.mask
                tmpdat_ew_slice = ((tmp_data_inst_2_bl* tmp_ew_wgt).sum(axis = 1)/tmp_ew_wgt.sum(axis = 0)).T


            tmpdat_ew_gdept=grid_dict[tmp_datstr]['gdept'][:,ew_jj_2nd_ind,ew_ii_2nd_ind].T
            ew_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0::2])*np.ma.masked
            for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ew_slice,tmpdat_ew_gdept,ew_slice_dict['y'].T)):ew_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))

    return ew_slice_dict

def reload_ns_data_comb(ii,jj,ti,thd, data_inst, nav_lon, nav_lat, grid_dict, regrid_meth,iijj_ind,load_2nd_files,configd):        
    tmp_datstr_mat = [ss for ss in data_inst.keys()]      
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')  
    '''
    reload the data for the N-S cross-section

    '''




    ns_slice_dict = {}
   
    ns_slice_dict['x'] =  nav_lat[:,ii]
    ns_slice_dict['y'] =  grid_dict['Dataset 1']['gdept'][:,:,ii]


    ns_slice_dict['Dataset 1'] = np.ma.masked_invalid(data_inst['Dataset 1'][:,:,ii])

    for tmp_datstr in tmp_datstr_mat_secondary:


        ns_ii_2nd_ind,ns_jj_2nd_ind,ns_bl_jj_ind_final,ns_bl_ii_ind_final,ns_wgt = iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'], iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]   ['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_wgt']



        if configd[2] is None:
            ns_slice_dict[tmp_datstr] = np.ma.masked_invalid(data_inst[tmp_datstr][:,:,ii])
        else:
            
            if regrid_meth == 1:
                tmpdat_ns_slice = np.ma.masked_invalid(data_inst[tmp_datstr][:,ns_jj_2nd_ind,ns_ii_2nd_ind].T)
            elif regrid_meth == 2:
                tmp_data_inst_2_bl = data_inst[tmp_datstr][:,ns_bl_jj_ind_final,ns_bl_ii_ind_final]
                tmp_ns_wgt = ns_wgt.copy()
                tmp_ns_wgt.mask = tmp_ns_wgt.mask | tmp_data_inst_2_bl.mask
                tmpdat_ns_slice = ((tmp_data_inst_2_bl* tmp_ns_wgt).sum(axis = 1)/tmp_ns_wgt.sum(axis = 0)).T

            tmpdat_ns_gdept = grid_dict[tmp_datstr]['gdept'][:,ns_jj_2nd_ind,ns_ii_2nd_ind].T
            ns_slice_dict[tmp_datstr] = np.ma.zeros(data_inst['Dataset 1'].shape[0:2])*np.ma.masked
            for i_i,(tmpdat,tmpz,tmpzorig) in enumerate(zip(tmpdat_ns_slice,tmpdat_ns_gdept,ns_slice_dict['y'].T)):ns_slice_dict[tmp_datstr][:,i_i] = np.ma.masked_invalid(np.interp(tmpzorig, tmpz, np.ma.array(tmpdat.copy(),fill_value=np.nan).filled()  ))


    return ns_slice_dict



            

def reload_hov_data_comb(var,var_mat,var_grid,deriv_var,ldi,thd,time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict, load_2nd_files,configd):       
    tmp_datstr_mat = [ss for ss in xarr_dict.keys()]      
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')    
            
    '''
    reload the data for the Hovmuller plot
    '''

    hov_dat = {}

    hov_dat['x'] = time_datetime
    hov_dat['y'] = grid_dict['Dataset 1']['gdept'][:,jj,ii]



    hov_x = time_datetime
    hov_y =  grid_dict['Dataset 1']['gdept'][:,jj,ii]

    hov_start = datetime.now()


    if var in deriv_var:
        #pdb.set_trace()
        if var == 'baroc_mag':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'

            hov_dat_U_dict = reload_hov_data_comb(tmp_var_U,var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
            hov_dat_V_dict = reload_hov_data_comb(tmp_var_V,var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)

            for tmp_datstr in tmp_datstr_mat: hov_dat[tmp_datstr]  = np.sqrt(hov_dat_U_dict[tmp_datstr]**2 + hov_dat_V_dict[tmp_datstr]**2)

       
        elif var == 'rho':
            hov_dat_T_dict = reload_hov_data_comb('votemper',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
            hov_dat_S_dict = reload_hov_data_comb('vosaline',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)

            
            for tmp_datstr in tmp_datstr_mat: hov_dat[tmp_datstr]  = sw_dens(hov_dat_T_dict[tmp_datstr], hov_dat_S_dict[tmp_datstr])

        else:
            for tmp_datstr in tmp_datstr_mat: hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked

    elif var in var_mat:


        hov_dat['Dataset 1'] = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']][:,:,jj,ii].load()).T
        
        for tmp_datstr in tmp_datstr_mat_secondary:
            
            ii_2nd_ind,jj_2nd_ind = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']
            if configd[2] is None:
                hov_dat['Dataset 2'] = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj,ii].load()).T
            else:
                hov_dat['Dataset 2'] = np.ma.zeros(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].shape[1::-1])*np.ma.masked
                tmpdat_hov = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj_2nd_ind,ii_2nd_ind].load())
                tmpdat_hov_gdept =  grid_dict[tmp_datstr]['gdept'][:,jj_2nd_ind,ii_2nd_ind]               
                for i_i,(tmpdat) in enumerate(tmpdat_hov):hov_dat['Dataset 2'][:,i_i] = np.ma.masked_invalid(np.interp(hov_dat['y'], tmpdat_hov_gdept, tmpdat))

    else:
        for tmp_datstr in tmp_datstr_mat: hov_dat[tmp_datstr] = np.ma.zeros((nz,ntime))*np.ma.masked
        
    hov_stop = datetime.now()
    return hov_dat

   
            
'''
def reload_hov_data_comb(var,var_mat,var_grid,deriv_var,ldi,thd,time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict, load_2nd_files,configd):    
            
    
    #reload the data for the Hovmuller plot
    
    if load_2nd_files:
        ii_2nd_ind,jj_2nd_ind = iijj_ind['Dataset 2']['ii'],iijj_ind['Dataset 2']['jj']
    hov_x = time_datetime
    hov_y =  grid_dict['Dataset 1']['gdept'][:,jj,ii]

    hov_start = datetime.now()


    if var in deriv_var:
        #pdb.set_trace()
        if var == 'baroc_mag':

            tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'
            hov_dat_U_1,hov_dat_U_2,hov_U_x,hov_U_y = reload_hov_data_comb(tmp_var_U,var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
            hov_dat_V_1,hov_dat_V_2,hov_V_x,hov_V_y = reload_hov_data_comb(tmp_var_V,var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)

            hov_dat_1 = np.sqrt(hov_dat_U_1**2 + hov_dat_V_1**2)
            hov_dat_2 = np.sqrt(hov_dat_U_2**2 + hov_dat_V_2**2)

       
        elif var == 'rho':
            hov_dat_T_1,hov_dat_T_2,hov_T_x,hov_T_y = reload_hov_data_comb('votemper',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
            hov_dat_S_1,hov_dat_S_2,hov_S_x,hov_S_y = reload_hov_data_comb('vosaline',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)

            hov_dat_1 = sw_dens(hov_dat_T_1, hov_dat_S_1)
            hov_dat_2 = sw_dens(hov_dat_T_2, hov_dat_S_2)

        else:
            hov_dat_1 = np.ma.zeros((nz,ntime))*np.ma.masked
            hov_dat_2 = np.ma.zeros((nz,ntime))*np.ma.masked

    elif var in var_mat:
        hov_dat_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']][:,:,jj,ii].load()).T
        
        if load_2nd_files:
            if configd[2] is None:
                hov_dat_2 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj,ii].load()).T
            else:
                hov_dat_2 = np.ma.zeros(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].shape[1::-1])*np.ma.masked
                tmpdat_hov = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][:,:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,:,jj_2nd_ind,ii_2nd_ind].load())
                tmpdat_hov_gdept =  grid_dict['Dataset 2']['gdept'][:,jj_2nd_ind,ii_2nd_ind]               
                for i_i,(tmpdat) in enumerate(tmpdat_hov):hov_dat_2[:,i_i] = np.ma.masked_invalid(np.interp(hov_y, tmpdat_hov_gdept, tmpdat))
        else:
            hov_dat_2 = hov_dat_1
    else:
        hov_dat_1 = np.ma.zeros((nz,ntime))*np.ma.masked
        hov_dat_2 = np.ma.zeros((nz,ntime))*np.ma.masked
        
    hov_stop = datetime.now()
    hov_dat = {}
    hov_dat['Dataset 1'] = hov_dat_1
    hov_dat['Dataset 2'] = hov_dat_2
    hov_dat['x'] = hov_x
    hov_dat['y'] = hov_y

    return hov_dat
'''




def reload_ts_data_comb(var,var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,z_meth,zz,zi,xarr_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,load_2nd_files):
        
    tmp_datstr_mat = [ss for ss in hov_dat_dict.keys()]      
    tmp_datstr_mat.remove('x')       
    tmp_datstr_mat.remove('y')    
    tmp_datstr_mat_secondary = tmp_datstr_mat.copy()
    if 'Dataset 1' in tmp_datstr_mat_secondary: tmp_datstr_mat_secondary.remove('Dataset 1')    
    #        
    ts_dat_dict = {}
    ts_dat_dict['x'] = time_datetime
    #for tmp_datstr in tmp_datstr_mat:

    if var_dim[var] == 3:
        if var in deriv_var:
            if var.upper() in ['PEA', 'PEAT','PEAS']:
               
                hov_dat_T_dict = reload_hov_data_comb('votemper',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
                hov_dat_S_dict = reload_hov_data_comb('vosaline',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)


                gdept_mat = grid_dict['Dataset 1']['gdept'][:,jj,ii]
                dz_mat = grid_dict['Dataset 1']['e3t'][:,jj,ii]
                nt_pea = hov_dat_T_dict['Dataset 1'].T.shape[0]

                gdept_mat_pea = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_pea)).T
                dz_mat_pea = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_pea)).T

                tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(hov_dat_T_dict['Dataset 1'].T[:,:,np.newaxis,np.newaxis],hov_dat_T_dict['Dataset 1'].T[:,:,np.newaxis,np.newaxis],gdept_mat_pea,dz_mat_pea,calc_TS_comp = True )
                if var.upper() == 'PEA':
                    ts_dat_dict['Dataset 1'] = tmppea_1[:,0,0] 
                elif var.upper() == 'PEAT':
                    ts_dat_dict['Dataset 1'] = tmppeat_1[:,0,0] 
                elif var.upper() == 'PEAS':
                    ts_dat_dict['Dataset 1'] = tmppeas_1[:,0,0] 

                for tmp_datstr in tmp_datstr_mat_secondary:
                    th_d_ind = int(tmp_datstr[-1])

                    tmppea_2, tmppeat_2, tmppeas_2 = pea_TS(hov_dat_T_dict[tmp_datstr].T[:,:,np.newaxis,np.newaxis],hov_dat_S_dict[tmp_datstr].T[:,:,np.newaxis,np.newaxis],gdept_mat_pea,dz_mat_pea,calc_TS_comp = True )

                    if var.upper() == 'PEA':
                        ts_dat_dict[tmp_datstr] = tmppea_2[:,0,0] 
                    elif var.upper() == 'PEAT':
                        ts_dat_dict[tmp_datstr] = tmppeat_2[:,0,0] 
                    elif var.upper() == 'PEAS':
                        ts_dat_dict[tmp_datstr] = tmppeas_2[:,0,0]
            else:

                for tmp_datstr in tmp_datstr_mat: ts_dat_dict[tmp_datstr] = np.ma.zeros((ntime))*np.ma.masked
                

        else:
            ts_dat_dict['Dataset 1'] = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']][:,jj,ii].load())
            
            for tmp_datstr in tmp_datstr_mat_secondary:
                th_d_ind = int(tmp_datstr[-1])
                ii_2nd_ind,jj_2nd_ind = iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj']
                if configd[th_d_ind] is None:
                    ts_dat_dict[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj,ii].load())
                else:
                    ts_dat_dict[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][var_grid[var]][ldi].variables[var][:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj_2nd_ind,ii_2nd_ind].load())




    elif var_dim[var] == 4:

        if z_meth in ['ss','nb','df','zm']:


            
            for tmp_datstr in tmp_datstr_mat:

                ss_ts_dat_1 = hov_dat_dict[tmp_datstr][0,:].ravel()
                hov_nb_ind_1 = (hov_dat_dict[tmp_datstr][:,0].mask == False).sum()-1
                nb_ts_dat_1 = hov_dat_dict[tmp_datstr][hov_nb_ind_1,:].ravel()
                df_ts_dat_1 = ss_ts_dat_1 - hov_nb_ind_1

                
                
                if z_meth == 'ss':
                    ts_dat_dict[tmp_datstr] = ss_ts_dat_1
                if z_meth == 'nb':
                    ts_dat_dict[tmp_datstr] = nb_ts_dat_1
                if z_meth == 'df':
                    ts_dat_dict[tmp_datstr] = df_ts_dat_1
                if z_meth == 'zm':
                    ts_e3t_1 = np.ma.array(grid_dict[tmp_datstr]['e3t'][:,jj,ii], mask = hov_dat_dict[tmp_datstr][:,0].mask)
                    ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum()
                    ts_dat_dict[tmp_datstr] = ((hov_dat_dict[tmp_datstr].T*ts_dm_wgt_1).T).sum(axis = 0)


        elif z_meth == 'z_slice':
            #pdb.set_trace()
            for tmp_datstr in tmp_datstr_mat:
                hov_zi = (np.abs(zz - hov_dat_dict['y'])).argmin()
                ts_dat_dict[tmp_datstr] = hov_dat_dict[tmp_datstr][hov_zi,:].ravel()


        elif z_meth == 'z_index':
            for tmp_datstr in tmp_datstr_mat:

                ts_dat_dict[tmp_datstr] = hov_dat_dict[tmp_datstr][zi,:]

    return ts_dat_dict 


"""
def reload_ts_data_comb(var,var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,z_meth,zz,xarr_dict,grid_dict,thd,var_mat,deriv_var,nz,ntime,configd,load_2nd_files):
    ts_x = time_datetime

    hov_y =  hov_dat_dict['y']
    hov_dat_1 = hov_dat_dict['Dataset 1']
    if load_2nd_files:
        hov_dat_2 = hov_dat_dict['Dataset 2']
    else:
        hov_dat_2 = hov_dat_1

    if load_2nd_files:
        ii_2nd_ind,jj_2nd_ind = iijj_ind['Dataset 2']['ii'],iijj_ind['Dataset 2']['jj']

    if var_dim[var] == 3:

        if var.upper() in ['PEA', 'PEAT','PEAS']:
           
            hov_dat_T_1,hov_dat_T_2,hov_T_x,hov_T_y = reload_hov_data_comb('votemper',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)
            hov_dat_S_1,hov_dat_S_2,hov_S_x,hov_S_y = reload_hov_data_comb('vosaline',var_mat,var_grid,deriv_var,ldi,thd, time_datetime, ii,jj,ii_2nd_ind,jj_2nd_ind,nz,ntime, grid_dict,xarr_dict,load_2nd_files,configd)

            gdept_mat = grid_dict['Dataset 1']['gdept'][:,jj,ii]
            dz_mat = grid_dict['Dataset 1']['e3t'][:,jj,ii]
            nt_pea = hov_dat_T_1.T.shape[0]

            gdept_mat_pea = np.tile(gdept_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_pea)).T
            dz_mat_pea = np.tile(dz_mat[np.newaxis,:,np.newaxis,np.newaxis].T,(1,1,1,nt_pea)).T

            tmppea_1, tmppeat_1, tmppeas_1 = pea_TS(hov_dat_T_1.T[:,:,np.newaxis,np.newaxis],hov_dat_S_1.T[:,:,np.newaxis,np.newaxis],gdept_mat_pea,dz_mat_pea,calc_TS_comp = True )
            if var.upper() == 'PEA':
                ts_dat_1 = tmppea_1[:,0,0] 
            elif var.upper() == 'PEAT':
                ts_dat_1 = tmppeat_1[:,0,0] 
            elif var.upper() == 'PEAS':
                ts_dat_1 = tmppeas_1[:,0,0] 

            if load_2nd_files:

                tmppea_2, tmppeat_2, tmppeas_2 = pea_TS(hov_dat_T_2.T[:,:,np.newaxis,np.newaxis],hov_dat_S_2.T[:,:,np.newaxis,np.newaxis],gdept_mat_pea,dz_mat_pea,calc_TS_comp = True )

                if var.upper() == 'PEA':
                    ts_dat_2 = tmppea_2[:,0,0] 
                elif var.upper() == 'PEAT':
                    ts_dat_2 = tmppeat_2[:,0,0] 
                elif var.upper() == 'PEAS':
                    ts_dat_2 = tmppeas_2[:,0,0] 

            else:
                ts_dat_2 = ts_dat_1#

        else:
            ts_dat_1 = np.ma.masked_invalid(xarr_dict['Dataset 1'][var_grid[var]][ldi].variables[var][:,thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']][:,jj,ii].load())
            if load_2nd_files:
                if configd[2] is None:
                    ts_dat_2 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,jj,ii].load())
                else:
                    ts_dat_2 = np.ma.masked_invalid(xarr_dict['Dataset 2'][var_grid[var]][ldi].variables[var][:,thd[2]['y0']:thd[2]['y1']:thd[2]['dy'],thd[2]['x0']:thd[2]['x1']:thd[2]['dx']][:,jj_2nd_ind,ii_2nd_ind].load())
            else:
                ts_dat_2 = ts_dat_1
    elif var_dim[var] == 4:

        if z_meth in ['ss','nb','df','zm']:

            ss_ts_dat_1 = hov_dat_1[0,:].ravel()
            hov_nb_ind_1 = (hov_dat_1[:,0].mask == False).sum()-1
            nb_ts_dat_1 = hov_dat_1[hov_nb_ind_1,:].ravel()
            df_ts_dat_1 = ss_ts_dat_1 - nb_ts_dat_1

            
            ss_ts_dat_2 = hov_dat_2[0,:].ravel()
            hov_nb_ind_2 = (hov_dat_2[:,0].mask == False).sum()-1
            nb_ts_dat_2 = hov_dat_2[hov_nb_ind_2,:].ravel()
            df_ts_dat_2 = ss_ts_dat_2 - nb_ts_dat_2
            
            
            if z_meth == 'ss':
                ts_dat_1 = ss_ts_dat_1
                ts_dat_2 = ss_ts_dat_2
            if z_meth == 'nb':
                ts_dat_1 = nb_ts_dat_1
                ts_dat_2 = nb_ts_dat_2
            if z_meth == 'df':
                ts_dat_1 = df_ts_dat_1
                ts_dat_2 = df_ts_dat_2
            if z_meth == 'zm':
                ts_e3t_1 = np.ma.array(grid_dict['Dataset 1']['e3t'][:,jj,ii], mask = hov_dat_2[:,0].mask)
                ts_dm_wgt_1 = ts_e3t_1/ts_e3t_1.sum()
                ts_dat_1 = ((hov_dat_1.T*ts_dm_wgt_1).T).sum(axis = 0)
                
                if load_2nd_files: # e3t_2nd only loaded if 2nd files present
                    ts_e3t_2 = np.ma.array(grid_dict['Dataset 2']['e3t'][:,jj_2nd_ind,ii_2nd_ind], mask = hov_dat_2[:,0].mask)
                    ts_dm_wgt_2 = ts_e3t_2/ts_e3t_2.sum()
                    ts_dat_2 = ((hov_dat_2.T*ts_dm_wgt_2).T).sum(axis = 0)
                else:
                    ts_dat_2 = ts_dat_1
        elif z_meth == 'z_slice':
            hov_zi = (np.abs(zz - hov_y)).argmin()
            ts_dat_1 = hov_dat_1[hov_zi,:].ravel()
            ts_dat_2 = hov_dat_2[hov_zi,:].ravel()


        elif z_meth == 'z_index':

            ts_dat_1 = hov_dat_1[zi,:]
            ts_dat_2 = hov_dat_2[zi,:]

    ts_dat = {}
    ts_dat['Dataset 1'] = ts_dat_1
    ts_dat['Dataset 2'] = ts_dat_2
    ts_dat['x'] = ts_x
    return ts_dat 

"""
def regrid_2nd(regrid_params,regrid_meth,thd,configd,dat_in): #):
    start_regrid_timer = datetime.now()


    if configd[2] is None:
        dat_out = dat_in
    else:
        (NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out) = regrid_params
        if (thd[1]['x0']!=0)|(thd[1]['y0']!=0): 
            print('thin_x0 and thin_y0 must equal 0, if not, need to work out thinning code in the regrid index method')
            pdb.set_trace()


        if regrid_meth == 1:
            # Nearest Neighbour Interpolation   ~0.01 sec
            #dat_out = dat_in[NWS_amm_nn_jj_ind_final,NWS_amm_nn_ii_ind_final]
            dat_out = dat_in[NWS_amm_nn_jj_ind_out,NWS_amm_nn_ii_ind_out]
            dat_out.mask = dat_out.mask|NWS_amm_wgt_out.mask.sum(axis =0)

        elif regrid_meth == 2:
            # Bilinear Interpolation            ~0.2sec

            dat_in_selected_corners =  dat_in[NWS_amm_bl_jj_ind_out ,NWS_amm_bl_ii_ind_out ].copy()
            NWS_amm_wgt_out.mask = NWS_amm_wgt_out.mask | dat_in_selected_corners.mask

            dat_out = (dat_in_selected_corners*NWS_amm_wgt_out).sum(axis = 0)/(NWS_amm_wgt_out).sum(axis = 0)
            

        else:
            print('config and configd[2] must be AMM15 and AMM7')
            pdb.set_trace()
    
    #if verbose_debugging:  print ('Regrid timer for method #%i: '%regrid_meth, datetime.now() - start_regrid_timer)
    return dat_out










def grad_horiz_ns_data(thd,grid_dict,ii, ns_slice_dat_1,ns_slice_dat_2):
    ns_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e2t'][2:,ii] +  grid_dict['Dataset 1']['e2t'][:-2,ii])/2 + grid_dict['Dataset 1']['e2t'][1:-1,ii])
    dns_1 = ns_slice_dat_1[:,2:] - ns_slice_dat_1[:,:-2]
    dns_2 = ns_slice_dat_2[:,2:] - ns_slice_dat_2[:,:-2]


    ns_slice_dat_1[:,1:-1] = dns_1/ns_slice_dx#_1
    ns_slice_dat_2[:,1:-1] = dns_2/ns_slice_dx#_2

    ns_slice_dat_1[:,0] = np.ma.masked
    ns_slice_dat_1[:,-1] = np.ma.masked
    ns_slice_dat_2[:,0] = np.ma.masked
    ns_slice_dat_2[:,-1] = np.ma.masked

    return ns_slice_dat_1,ns_slice_dat_2


def grad_horiz_ew_data(thd,grid_dict,jj, ew_slice_dat_1,ew_slice_dat_2):
    ew_slice_dx =  thd[1]['dx']*((grid_dict['Dataset 1']['e1t'][jj,2:] +  grid_dict['Dataset 1']['e1t'][jj,:-2])/2 + grid_dict['Dataset 1']['e1t'][jj,1:-1])
    dew_1 = ew_slice_dat_1[:,2:] - ew_slice_dat_1[:,:-2]
    dew_2 = ew_slice_dat_2[:,2:] - ew_slice_dat_2[:,:-2]

    ew_slice_dat_1[:,1:-1] = dew_1/ew_slice_dx#_1
    ew_slice_dat_2[:,1:-1] = dew_2/ew_slice_dx#_2

    ew_slice_dat_1[:,0] = np.ma.masked
    ew_slice_dat_1[:,-1] = np.ma.masked
    ew_slice_dat_2[:,0] = np.ma.masked
    ew_slice_dat_2[:,-1] = np.ma.masked

    return ew_slice_dat_1, ew_slice_dat_2


def grad_vert_ns_data(ns_slice_dat_1,ns_slice_dat_2,ns_slice_y):
    dns_1 = ns_slice_dat_1[2:,:] - ns_slice_dat_1[:-2,:]
    dns_2 = ns_slice_dat_2[2:,:] - ns_slice_dat_2[:-2,:]
    dns_z = ns_slice_y[2:,:] - ns_slice_y[:-2,:]




    ns_slice_dat_1[1:-1,:] = dns_1/dns_z
    ns_slice_dat_2[1:-1,:] = dns_2/dns_z
    ns_slice_dat_1[ 0,:] = np.ma.masked
    ns_slice_dat_1[-1,:] = np.ma.masked
    ns_slice_dat_2[ 0,:] = np.ma.masked
    ns_slice_dat_2[-1,:] = np.ma.masked

    return ns_slice_dat_1,ns_slice_dat_2


def grad_vert_ew_data(ew_slice_dat_1,ew_slice_dat_2,ew_slice_y):
    dew_1 = ew_slice_dat_1[2:,:] - ew_slice_dat_1[:-2,:]
    dew_2 = ew_slice_dat_2[2:,:] - ew_slice_dat_2[:-2,:]
    dew_z = ew_slice_y[2:,:] - ew_slice_y[:-2,:]

    ew_slice_dat_1[1:-1,:] = dew_1/dew_z
    ew_slice_dat_2[1:-1,:] = dew_2/dew_z

    ew_slice_dat_1[ 0,:] = np.ma.masked
    ew_slice_dat_1[-1,:] = np.ma.masked
    ew_slice_dat_2[ 0,:] = np.ma.masked
    ew_slice_dat_2[-1,:] = np.ma.masked

    return ew_slice_dat_1, ew_slice_dat_2


#def grad_vert_hov_data(hov_dat_1,hov_dat_2,hov_y):
def grad_vert_hov_data(hov_dat_dict):

    #hov_dat_dict['Dataset 1'],hov_dat_dict['Dataset 2'] = grad_vert_hov_data(hov_dat_dict['Dataset 1'],hov_dat_dict['Dataset 2'],hov_dat_dict['y'])

    tmp_datstr_mat = [ss for ss in hov_dat_dict.keys()]


    hov_y = hov_dat_dict['y']
    dhov_z = hov_y[2:] - hov_y[:-2]
    
    for tmp_datstr in tmp_datstr_mat:

        dhov_1 = hov_dat_dict[tmp_datstr][2:,:] - hov_dat_dict[tmp_datstr][:-2,:]
        hov_dat_dict[tmp_datstr][1:-1,:] = (dhov_1.T/dhov_z).T
        hov_dat_dict[tmp_datstr][ 0,:] = np.ma.masked
        hov_dat_dict[tmp_datstr][-1,:] = np.ma.masked


    return hov_dat_dict


def extract_time_from_xarr(xarr_dict_in,ex_fname_in,time_varname,t_dim,date_in_ind,date_fmt,ti,verbose_debugging):

    '''
    
    time_datetime,time_datetime_since_1970,ntime = extract_time_from_xarr(xarr_dict['Dataset 1']['T'],fname_dict['Dataset 1']['T'][0],date_in_ind,date_fmt,verbose_debugging)
    '''
    #pdb.set_trace()
    
    print ('xarray start reading nctime',datetime.now())
    nctime = xarr_dict_in[0].variables[time_varname]

    try:
        
        if 'time_origin' in nctime.attrs.keys():
            nc_time_origin = nc_time_var.time_origin
        else:
            nc_time_origin = '1980-01-01 00:00:00'
            print('No time origin set - set to 1/1/1980. Other Time parameters likely to be missing')
    except:
        print('Except: extract_time_from_xarr, couldn''t to read time_origin from xarray, using netCDF4')
        rootgrp_hpc_time = Dataset(ex_fname_in, 'r', format='NETCDF4')
        
        nc_time_var = rootgrp_hpc_time.variables[time_varname]
        if 'time_origin' in nc_time_var.ncattrs():
            nc_time_origin = nc_time_var.time_origin
        else:
            nc_time_origin = '1980-01-01 00:00:00'
            print('No time origin set - set to 1/1/1980. Other Time parameters likely to be missing')

            #pdb.set_trace()
        rootgrp_hpc_time.close()




    #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
    nctime_calendar_type = None
    try:
        if 'calendar' in nctime.attrs.keys():
            nctime_calendar_type = nc_time_var.calendar
        else: 
            print('calendar not in time info')
    except:
        nctime_calendar_type = None
    

    if nctime_calendar_type is None:
        if type(np.array(nctime)[0]) is type(cftime._cftime.Datetime360Day(1980,1,1)):
            nctime_calendar_type = '360'
        else:
            nctime_calendar_type = 'greg'


    try:


        #different treatment for 360 days and gregorian calendars... needs time_datetime for plotting, and time_datetime_since_1970 for index selection
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


        if date_in_ind is not None:
            date_in_ind_datetime = datetime.strptime(date_in_ind,date_fmt)
            date_in_ind_datetime_timedelta = np.array([(ss - date_in_ind_datetime).total_seconds() for ss in time_datetime])
            ti = np.abs(date_in_ind_datetime_timedelta).argmin()
            if verbose_debugging: print('Setting ti from date_in_ind (%s): ti = %i (%s). '%(date_in_ind,ti, time_datetime[ti]), datetime.now())

    except:
        print()
        print()
        print()
        print(' Not able to read time in second data set, using dummy time')
        print()
        print()
        print()
        time_datetime = np.array([datetime(datetime.now().year, datetime.now().month, datetime.now().day) + timedelta(days = i_i) for i_i in range( xarr_dict_in[0][0].dims[t_dim])])
        time_datetime_since_1970 = np.array([(ss - datetime(1970,1,1,0,0)).total_seconds()/86400 for ss in time_datetime])

        if date_in_ind is not None: ti = 0
    ntime = time_datetime.size

    return time_datetime,time_datetime_since_1970,ntime,ti




if __name__ == "__main__":
    main()
