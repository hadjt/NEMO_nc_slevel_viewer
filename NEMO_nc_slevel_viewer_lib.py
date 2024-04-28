
import pdb,sys,os

from datetime import datetime, timedelta

from netCDF4 import Dataset,num2date


import numpy as np



import matplotlib.pyplot as plt

#from python3_plotting_function import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region


from matplotlib.colors import LinearSegmentedColormap, ListedColormap


import socket
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

    poss_zdims = ['deptht','depthu','depthv']
    poss_tdims = ['time_counter','time']
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


    print ('Think this is simpler with Python3')
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




def pea_TS(T_in,S_in,gdept,e3t_in,tmask,calc_TS_comp = False, zcutoff = 400.):
    #from call_eos import calc_sigma0, calc_sigmai#, calc_albet

    # Create potential energy anomaly.

    nt,nz = T_in.shape[:2]
    # if t and s are not masked arrays turn them into them
    if np.ma.isMA(T_in):
        T = T_in.copy()
        S = S_in.copy()
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
    from NEMO_nc_slevel_viewer_lib import sw_dens
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
    


if __name__ == "__main__":
    main()
