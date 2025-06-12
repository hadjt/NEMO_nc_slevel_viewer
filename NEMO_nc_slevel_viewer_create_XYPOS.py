#salloc --mem=24000 --time=360 

import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
from netCDF4 import Dataset,stringtochar, chartostring #,date2num,num2date
import pdb,os,sys
import os.path
#import xarray
#import glob
#import matplotlib

#import time
#import argparse
#import textwrap
#import psutil

#from scipy.interpolate import griddata




from NEMO_nc_slevel_viewer_lib import int_ind_wgt_from_xypos, ind_from_lon_lat,int_ind_wgt_from_xypos,int_ind_wgt_from_xypos_func,load_xypos,cut_down_xypos

def create_xypos(mesh_file,xypos_file_out, xypos_file = None, DLONLAT = 0.1,
    LAT_min = None,
    LAT_max = None, 
    LON_min = None,
    LON_max = None,
    DLAT = None,
    DLON = None,
    nav_lon = None,
    nav_lat = None,
    n_cutout_size = 15):

    if (nav_lon is None) & (nav_lat is None):
    # Read lon and lat data from mesh file
        rootgrp = Dataset(mesh_file, 'r')
        nav_lon = rootgrp.variables['glamt'][0,:,:]
        nav_lat = rootgrp.variables['gphit'][0,:,:]
        rootgrp.close()

    # extract shape, and min and max values
    n_nav_lat, n_nav_lon = nav_lon.shape
    nav_lon_min = nav_lon.min()
    nav_lon_max = nav_lon.max()
    nav_lat_min = nav_lat.min()
    nav_lat_max = nav_lat.max()

    # if min, max lon and lat, and dlon, dlat, not specified, caluclate
    if DLON is None: DLON = DLONLAT
    if DLAT is None: DLAT = DLONLAT
    
    if LAT_min is None: LAT_min = np.floor(nav_lat_min/DLAT)*DLAT
    if LAT_max is None: LAT_max = np.ceil(nav_lat_max/DLAT)*DLAT
    if LON_min is None: LON_min = np.floor(nav_lon_min/DLON)*DLON
    if LON_max is None: LON_max = np.ceil(nav_lon_max/DLON)*DLON




    # Create regular lon and lat arrays for LUT
    xypos_lon_int = np.arange(LON_min,LON_max + DLON, DLON)
    xypos_lat_int = np.arange(LAT_min,LAT_max + DLAT, DLAT)
    xypos_lon_int_mat, xypos_lat_int_mat = np.meshgrid(xypos_lon_int,xypos_lat_int)

    # Find size of lon,lat LUT, and create index table for them. 
    nlon  = xypos_lon_int.size
    nlat  = xypos_lat_int.size
    xypos_ii_int = np.arange(nlon)
    xypos_jj_int = np.arange(nlat)
    xypos_ii_int_mat, xypos_jj_int_mat = np.meshgrid(xypos_ii_int,xypos_jj_int)




    # Find size of input model nav_lon,and nav_lat arrays , and create index table for them. 
    nav_ii_int = np.arange(n_nav_lon)
    nav_jj_int = np.arange(n_nav_lat)
    nav_ii_int_mat, nav_jj_int_mat = np.meshgrid(nav_ii_int,nav_jj_int)


    #preallocate x and y index tables, and distance, and argmin values. Also previous i and J
    xypos_dist_mat = np.ma.zeros(xypos_lon_int_mat.shape)*np.ma.masked
    xypos_arg_mat = np.ma.zeros(xypos_lon_int_mat.shape)*np.ma.masked
    xypos_X_int_mat = np.ma.zeros(xypos_lon_int_mat.shape, dtype = 'int')*np.ma.masked
    xypos_Y_int_mat = np.ma.zeros(xypos_lon_int_mat.shape, dtype = 'int')*np.ma.masked
    pvi_mat = np.ma.zeros(xypos_lon_int_mat.shape)
    pvj_mat = np.ma.zeros(xypos_lon_int_mat.shape)


    # Cycle through points of the lon and lat LUT, and find the distance to model Lon and Lat. 
    # Find the closest, and record it and its distance.
    # For the following grid boxes in the row, search a cut out domain around the previous point, to speed up calculation. 
    
    
    for xyp_i,xyp_lon in enumerate(xypos_lon_int):
        print('XYPOS', datetime.now(),xyp_i,len(xypos_lon_int))
        # Reset previous i and j ind at the beginning of each row to None
        pvi,pvj=None,None
        for xyp_j,xyp_lat in enumerate(xypos_lat_int):
            # if previous i and j is None, search whole domain
            if pvi is None:
                s_nav_lon,s_nav_lat,s_ii_mat, s_jj_mat = nav_lon,nav_lat,nav_ii_int_mat, nav_jj_int_mat
            else:
                #if previous i and j valid, use a cut out around that point 
                s_nav_lon,s_nav_lat,s_ii_mat,s_jj_mat = subset_mats(pvi,pvj,n_nav_lon,n_nav_lat,nav_lon,nav_lat,nav_ii_int_mat, nav_jj_int_mat,n_cutout_size)
                '''
                plt.plot(s_nav_lon.ravel(),s_nav_lat.ravel(),'x')
                plt.plot(xyp_lon,xyp_lat,'+')
                plt.show()
                '''
                #pdb.set_trace()


            tmpxydist = np.sqrt((s_nav_lon - xyp_lon)**2 + (s_nav_lat - xyp_lat)**2).ravel()
            tmpxydist_min = tmpxydist.min()
            tmpxydist_argmin = tmpxydist.argmin()
            #xypos_X_int_mat[xyp_j,xyp_i] = int(s_ii_mat.ravel()[tmpxydist_argmin])
            #xypos_Y_int_mat[xyp_j,xyp_i] = int(s_jj_mat.ravel()[tmpxydist_argmin])
            xy_tmp_i =  int(s_ii_mat.ravel()[tmpxydist_argmin])
            xy_tmp_j =  int(s_jj_mat.ravel()[tmpxydist_argmin])

            xypos_X_int_mat[xyp_j,xyp_i] = int(xy_tmp_i)
            xypos_Y_int_mat[xyp_j,xyp_i] = int(xy_tmp_j)
            xypos_dist_mat[xyp_j,xyp_i] = tmpxydist_min
            xypos_arg_mat[xyp_j,xyp_i] = tmpxydist_argmin

            pvi,pvj=xy_tmp_i,xy_tmp_j#int(xypos_X_int_mat[xyp_j,xyp_i]),int(xypos_Y_int_mat[xyp_j,xyp_i])

            #print(xyp_lon,xyp_lat)
            #print(s_nav_lon[xy_tmp_j,xy_tmp_i],s_nav_lat[xy_tmp_j,xy_tmp_i])
            #print(xyp_j,xyp_i)
            if tmpxydist_min>DLAT:#4*((np.sqrt(DLAT**2 + DLON**2))):
                #pdb.set_trace()
                xypos_X_int_mat[xyp_j,xyp_i] = np.ma.masked
                xypos_Y_int_mat[xyp_j,xyp_i] = np.ma.masked
                pvi,pvj = None, None

            pvi_mat[xyp_j,xyp_i] = pvi
            pvj_mat[xyp_j,xyp_i] = pvj

            #pvi,pvj=xyp_i,xyp_j

            #pvi,pvj = None, None

            #if xyp_i == 10: pdb.set_trace()
            
            
            ''' 
            plt.pcolormesh()
            plt.colorbar()
            plt.show()


            '''


            '''
            tmpxydist = (nav_lon - xyp_lon)**2 + (nav_lat - xyp_lat)**2
            tmpxydist_min = tmpxydist.min()
            if tmpxydist_min<(np.sqrt(DLAT**2 + DLON**2)):

                tmpxydist_argmin = tmpxydist.argmin()
                xypos_X_int_mat[xyp_j,xyp_i] = int(tmpxydist_argmin%nlon)
                xypos_Y_int_mat[xyp_j,xyp_i] = int(tmpxydist_argmin//nlon)
                pvi,pvj=xyp_i,xyp_j
            '''
        #if (xyp_i > 50): pdb.set_trace()



    if xypos_file is not None:
        rootgrp = Dataset(xypos_file, 'r')
        XPOS_in = rootgrp.variables['XPOS'][:,:]
        YPOS_in = rootgrp.variables['YPOS'][:,:]
        LON_in = rootgrp.variables['LON'][:,:]
        LAT_in = rootgrp.variables['LAT'][:,:]
        rootgrp.close()


        #pdb.set_trace()
        #
        ax = [plt.subplot(2,2,1)]
        plt.pcolormesh(XPOS_in)
        plt.colorbar()
        ax.append(plt.subplot(2,2,2, sharex = ax[0], sharey = ax[0]))
        #plt.pcolormesh(xypos_X_int_mat)
        plt.pcolormesh(np.ma.array(xypos_X_int_mat, mask = xypos_dist_mat>DLAT))
        plt.colorbar()
        ax.append(plt.subplot(2,2,3, sharex = ax[0], sharey = ax[0]))
        plt.pcolormesh(XPOS_in[:192,:415] - xypos_X_int_mat[:192,:415])
        plt.colorbar()
        ax.append(plt.subplot(2,2,4, sharex = ax[0], sharey = ax[0]))
        plt.pcolormesh(xypos_dist_mat[:192,:415] )
        plt.colorbar()    
        plt.show()
        
        pdb.set_trace()
        #
    


    '''
[jonathan.tinker@cazldf0000ET NEMO_nc_slevel_viewer]$ ncdump -h /data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/xypos_amm15.nc
netcdf xypos_amm15 {
dimensions:
	nx = 416 ;
	ny = 192 ;
variables:
	float LON(ny, nx) ;
		LON:long_name = "longitude" ;
	float LAT(ny, nx) ;
		LAT:long_name = "latitude" ;
	int XPOS(ny, nx) ;
		XPOS:long_name = "xposition" ;
		XPOS:_FillValue = -1 ;
	int YPOS(ny, nx) ;
		YPOS:long_name = "yposition" ;
		YPOS:_FillValue = -1 ;

// global attributes:
		:title = "Mappingfilefromlon/lattomodelgridpoint" ;
		:maxxdiff = 7 ;
		:maxydiff = 9 ;
		:dlon = 0.1 ;
		:dlat = 0.1 ;
		:lonmin = -25.35 ;
		:latmin = 44.15 ;
}


    '''
    rootgrp_out = Dataset(xypos_file_out, 'w', format='NETCDF4')
    nx_dim = rootgrp_out.createDimension('nx', nlon)
    ny_dim = rootgrp_out.createDimension('ny', nlat)

    nc_2d_var_dict = {}
    for vv in ['LON','LAT']: nc_2d_var_dict[vv] = rootgrp_out.createVariable(vv,'f4',('ny','nx'), zlib=True)
    for vv in ['XPOS','YPOS']: nc_2d_var_dict[vv] = rootgrp_out.createVariable(vv,'i4',('ny','nx'),fill_value = -999., zlib=True)
    nc_2d_var_dict['LON'][:,:] = xypos_lon_int_mat
    nc_2d_var_dict['LAT'][:,:] = xypos_lat_int_mat
    nc_2d_var_dict['XPOS'][:,:] = xypos_X_int_mat
    nc_2d_var_dict['YPOS'][:,:] = xypos_Y_int_mat


    rootgrp_out.title = "Mappingfilefromlon/lattomodelgridpoint"
    rootgrp_out.note = "NEMO output emulated with NEMO_nc_slevel_viewer_create_XYPOS.py python code" 
    rootgrp_out.maxxdiff = 7
    rootgrp_out.maxydiff = 9
    rootgrp_out.dlon = DLON
    rootgrp_out.dlat = DLAT
    rootgrp_out.lonmin = LON_min
    rootgrp_out.latmin = LAT_min
    rootgrp_out.close()


    print ('Finished creating %s at %s'%(xypos_file_out, datetime.now()))
    pdb.set_trace()


def subset_mats(pvi,pvj,nlon,nlat,nav_lon,nav_lat,xypos_ii_int_mat, xypos_jj_int_mat,npnt):

    ii_min = np.minimum(np.maximum((pvi - npnt),0),nlon-1)
    jj_min = np.minimum(np.maximum((pvj - npnt),0),nlat-1)
    ii_max = np.minimum(np.maximum((pvi + npnt + 1),0),nlon-1)
    jj_max = np.minimum(np.maximum((pvj + npnt + 1),0),nlat-1)
    
    s_nav_lon = nav_lon[jj_min:jj_max+1,ii_min:ii_max+1]
    s_nav_lat = nav_lat[jj_min:jj_max+1,ii_min:ii_max+1]
    s_ii_mat = xypos_ii_int_mat[jj_min:jj_max+1,ii_min:ii_max+1]
    s_jj_mat = xypos_jj_int_mat[jj_min:jj_max+1,ii_min:ii_max+1]

    #pdb.set_trace()

    
    return s_nav_lon,s_nav_lat,s_ii_mat, s_jj_mat 
                

def create_OSTIA_xypos():
    xypos_file_out='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/OSTIA/xypos_OSTIA.nc'
    eg_ostia_file = '/data/users/ofrd-mopa/ostia/data/netcdf/2022/01/20220131120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc'
    
    rootgrp = Dataset(eg_ostia_file, 'r')
    ostia_lon = rootgrp.variables['lon'][:]
    ostia_lat = rootgrp.variables['lat'][:]
    rootgrp.close()
    DLAT = 0.1
    DLON = 0.1
    
    LAT_min = ostia_lat.min()#40
    LAT_max = ostia_lat.max()#70  
    LON_min = ostia_lon.min()#-30
    LON_max = ostia_lon.max()#20
    nav_lon, nav_lat = np.meshgrid(ostia_lon,ostia_lat)

    pdb.set_trace()

    #create_xypos(mesh_file,xypos_file_out,LAT_min = LAT_min,LAT_max = LAT_max,LON_min = LON_min,LON_max = LON_max,DLAT = DLAT,DLON = DLON)
    create_xypos('',xypos_file_out,LAT_min = LAT_min,LAT_max = LAT_max,LON_min = LON_min,LON_max = LON_max,DLAT = DLAT,DLON = DLON,nav_lon = nav_lon,nav_lat = nav_lat)
    
  
def create_test_amm15_xypos():
    mesh_file='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/amm15.mesh_mask.nc'
    xypos_file='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/xypos_amm15.nc'
    xypos_file_out='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/tmp_out_xypos_amm15.nc'

    #create_xypos(mesh_file,xypos_file_out,xypos_file=xypos_file,LAT_min = LAT_min,LAT_max = LAT_max,LON_min = LON_min,LON_max = LON_max,DLAT = DLAT,DLON = DLON)

    LAT_min = 44.15
    LAT_max = 63.25    
    LON_min = -25.35
    LON_max = 16.15
    DLAT = 0.1
    DLON = 0.1
    
    xypos_file_out='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/tmp_out_xypos_amm15__2.nc'
    #create_xypos(mesh_file,xypos_file_out,LAT_min = LAT_min,LAT_max = LAT_max,LON_min = LON_min,LON_max = LON_max,DLAT = DLAT,DLON = DLON)
    



def create_shmi_balmfc_xypos():
    mesh_file='/data/users/jonathan.tinker/shelf_seas/NEMO_nc_slevel_viewer_data/SMHI/mesh_mask.nc'
    xypos_file_out='/data/users/jonathan.tinker/shelf_seas/NEMO_nc_slevel_viewer_data/SMHI/tmp_out_xypos_BALMFCNRT.nc'
    create_xypos(mesh_file,xypos_file_out)


def create_bsh_nwsmfc_my_xypos():
 


    mesh_file='/data/users/jonathan.tinker/shelf_seas/NEMO_nc_slevel_viewer_data/BSH/domain_cfg.nc'
    xypos_file_out='/data/users/jonathan.tinker/shelf_seas/NEMO_nc_slevel_viewer_data/BSH/tmp_out_xypos_BSH_NWSMFC_MY.nc'
    create_xypos(mesh_file,xypos_file_out)



def test_xypos_interp():
    print('t1','%s'%datetime.now())

    xypos_fname='/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/OSTIA/xypos_OSTIA.nc'
    eg_ostia_file = '/data/users/ofrd-mopa/ostia/data/netcdf/2022/01/20220131120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc'
    mesh_file = '/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM15/amm15.mesh_mask.nc'
    mesh_file = '/data/users/jonathan.tinker/reffiles/NEMO_nc_slevel_viewer/AMM7/amm7.mesh_mask.nc'

    rootgrp = Dataset(eg_ostia_file, 'r')
    ostia_lon = rootgrp.variables['lon'][:]
    ostia_lat = rootgrp.variables['lat'][:]
    ostia_sst = rootgrp.variables['analysed_sst'][0,:]-273.15
    ostia_mask = rootgrp.variables['mask'][0,:]
    rootgrp.close()
    ostia_nav_lon, ostia_nav_lat = np.meshgrid(ostia_lon,ostia_lat)




    rootgrp = Dataset(mesh_file, 'r')
    nav_lon = rootgrp.variables['glamt'][0,:,:]
    nav_lat = rootgrp.variables['gphit'][0,:,:]
    rootgrp.close() 

    print('t2','%s'%datetime.now())

    xypos_dict = load_xypos(xypos_fname)
    print('t3','%s'%datetime.now())

    xypos_dict = cut_down_xypos(xypos_dict, nav_lon.min(), nav_lat.min(),nav_lon.max(), nav_lat.max())
    
    print('t4','%s'%datetime.now())

    sel_bl_jj_out, sel_bl_ii_out, NWS_wgt, sel_jj_out, sel_ii_out = int_ind_wgt_from_xypos_func(xypos_dict, nav_lon,nav_lat, ostia_nav_lon, ostia_nav_lat)
    
    print('t5','%s'%datetime.now())

    pdb.set_trace()
    ostia_nav_lon[sel_jj_out, sel_ii_out]

    plt.subplot(2,2,1)
    plt.pcolormesh(ostia_nav_lon[sel_jj_out, sel_ii_out] - nav_lon)#, vmin = -0.1, vmax = 0.1)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.pcolormesh((ostia_nav_lon[sel_bl_jj_out, sel_bl_ii_out]*NWS_wgt).sum(axis = 0) - nav_lon)#, vmin = -0.1, vmax = 0.1)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.pcolormesh(ostia_nav_lat[sel_jj_out, sel_ii_out] - nav_lat)#, vmin = -0.1, vmax = 0.1)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.pcolormesh((ostia_nav_lat[sel_bl_jj_out, sel_bl_ii_out]*NWS_wgt).sum(axis = 0) - nav_lat)#, vmin = -0.1, vmax = 0.1)
    plt.colorbar()
    plt.show()

        
    interp_SST = (ostia_sst[sel_bl_jj_out, sel_bl_ii_out]*NWS_wgt).sum(axis = 0)
    interp_mask = ((ostia_mask==1).astype('int')[sel_bl_jj_out,sel_bl_ii_out]*NWS_wgt).sum(axis = 0)      
    interp_SST[interp_mask!=1] = np.ma.masked
    plt.pcolormesh(interp_SST)
    plt.colorbar()
    plt.show()


    pdb.set_trace()


def main():


    pdb.set_trace()

    test_xypos_interp()

    pdb.set_trace()
    create_OSTIA_xypos()
    #create_test_amm15_xypos()
    #create_shmi_balmfc_xypos()
    create_bsh_nwsmfc_my_xypos()

if __name__ == "__main__":
    main()


