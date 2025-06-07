import matplotlib.pyplot as plt

from datetime import datetime,timedelta
import numpy as np
from netCDF4 import Dataset,stringtochar, chartostring #,date2num,num2date
import pdb,os,sys
import os.path
import xarray
import glob
#import cftime
import matplotlib
#import csv

import time
import argparse
import textwrap
import psutil

from scipy.interpolate import griddata

### set-up modules
from NEMO_nc_slevel_viewer_lib import create_config_fnames_dict,create_rootgrp_gdept_dict,create_gdept_ncvarnames
from NEMO_nc_slevel_viewer_lib import create_col_lst,create_Dataset_lst,create_xarr_dict
from NEMO_nc_slevel_viewer_lib import create_lon_lat_dict,create_ncvar_lon_lat_time_dict

from NEMO_nc_slevel_viewer_lib import trim_file_dict,remove_extra_end_file_dict,add_derived_vars
from NEMO_nc_slevel_viewer_lib import connect_to_files_with_xarray,load_grid_dict

from NEMO_nc_slevel_viewer_lib import extract_time_from_xarr,resample_xarray

# Data loading modules
from NEMO_nc_slevel_viewer_lib import reload_data_instances
from NEMO_nc_slevel_viewer_lib import reload_map_data_comb,reload_ew_data_comb,reload_ns_data_comb
from NEMO_nc_slevel_viewer_lib import reload_hov_data_comb,reload_ts_data_comb,reload_pf_data_comb


# Data processing modules
from NEMO_nc_slevel_viewer_lib import grad_horiz_ns_data,grad_horiz_ew_data
from NEMO_nc_slevel_viewer_lib import grad_vert_ns_data,grad_vert_ew_data,grad_vert_hov_prof_data

from NEMO_nc_slevel_viewer_lib import field_gradient_2d, vector_div, vector_curl,sw_dens

# Data manipulation modules
from NEMO_nc_slevel_viewer_lib import rotated_grid_from_amm15, reduce_rotamm15_grid,regrid_2nd_thin_params,regrid_iijj_ew_ns
from NEMO_nc_slevel_viewer_lib import interp1dmat_create_weight

# Plotting modules
from NEMO_nc_slevel_viewer_lib import get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region,get_colorbar_values
from NEMO_nc_slevel_viewer_lib import scale_color_map,lon_lat_to_str,current_barb,get_pnts_pcolor_in_region
from NEMO_nc_slevel_viewer_lib import profile_line

from NEMO_nc_slevel_viewer_lib import load_ops_prof_TS, load_ops_2D_xarray, obs_reset_sel
from NEMO_nc_slevel_viewer_lib import pop_up_opt_window,pop_up_info_window,get_help_text,jjii_from_lon_lat

from NEMO_nc_slevel_viewer_lib import calc_ens_stat_2d, calc_ens_stat_3d,calc_ens_stat_map

from NEMO_nc_slevel_viewer_lib import int_ind_wgt_from_xypos, ind_from_lon_lat


letter_mat = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

import socket
computername = socket.gethostname()
comp = 'linux'
if computername in ['xcel00','xcfl00']: comp = 'hpc'

import warnings
warnings.filterwarnings("ignore")

script_dir=os.path.dirname(os.path.realpath(__file__)) + '/'

global fname_lst, fname_lst_2nd,var

#import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

import matplotlib.patheffects as pe
#matplotlib.use('Qt5Agg')
#def mon_mean(x):
#    return x.groupby('time.month').mean('time')


def nemo_slice_zlev(config = 'amm7',  
    zlim_max = None,var = None,
    fig_lab_d = None,configd = None,thd = None,fname_dict = None,load_second_files = False,
    xlim = None, ylim = None, tlim = None, clim = None,
    ii = None, jj = None, ti = None, zz = None, zi = None, 
    lon_in = None, lat_in = None, date_in_ind = None, date_fmt = '%Y%m%d',
    cutxind = None, cutyind=None,
    z_meth = None,
    secdataset_proc = 'Dataset 1',
    hov_time = False, do_cont = False, do_grad = 0,
    allow_diff_time = False,
    preload_data = True,
    ld_lst = None, ld_nctvar = 'time_counter',ld_lab_lst = '-36,-12,012,036,060,084,108,132',
    clim_sym = None, clim_pair = True,use_cmocean = False,
    fig_dir = None,fig_lab = 'figs',fig_cutout = True, 
    justplot = False, justplot_date_ind = None,justplot_z_meth_zz = None,justplot_secdataset_proc = None,
    fig_fname_lab = None, fig_fname_lab_2nd = None,
    trim_files = True,
    trim_extra_files = False,
    vis_curr = -1, vis_curr_meth = 'barb',
    resample_freq = None,
    verbose_debugging = False,do_timer = True,do_memory = True,do_ensemble = False,
    Obs_dict = None, Obs_reloadmeth = 2,Obs_hide = False,
    do_MLD = True,do_mask = False,
    use_xarray_gdept = True,
    force_dim_d = None,xarr_rename_master_dict=None):

    print('Initialise at ',datetime.now())
    init_timer = []
    init_timer.append((datetime.now(),'Starting Program'))


    '''
    pdb.set_trace()
    screen = plt.get_current_fig_manager().window.screen()
    print(screen.size())
    >> PySide6.QtCore.QSize(1920, 969)  # size in pixels

    # Rough calculation of size of screen in inches
    print(screen.size() / screen.physicalDotsPerInch())

    pdb.set_trace()
    '''
    ## Old VDI figheight = 12
    ## Old VDI figwidth = 18
    figheight = 7
    figwidth = 15.3
    
    figsuptitfontsize = 14*0.8
    matplotlib.rcParams['font.size'] = 10*0.8

    cutout_data = False
    if cutxind is None:
        cutxind = [0,None]
    else:
        cutout_data = True
    if cutyind is None:
        cutyind = [0,None]
    else:
        cutout_data = True
    

    
    ens_stat_lst = ['EnsMean','EnsStd','EnsVar','EnsCnt']
    if do_ensemble:
        Ens_stat = None
    #    ens_stat_lst = ['EnsMean','EnsStd','EnsVar','EnsCnt']
    #else:
    #    ens_stat_lst = []

    # if arguement Obs_dict is not None, set do_Obs to true
    #Obs_dict = Obs_fname
    if Obs_dict is None:
        do_Obs = False
    else:
        do_Obs = True
        #pdb.set_trace()
        
        for tmp_datstr in Obs_dict.keys():
            for ob_var in Obs_dict[tmp_datstr].keys():
                Obs_dict_is_str = isinstance(Obs_dict[tmp_datstr][ob_var] , str)

        if Obs_dict_is_str== False:
            Obs_reloadmeth = -1
        elif Obs_dict_is_str:
            Obs_fname = Obs_dict

            

            Obs_dict = {}


            for tmp_datstr in Obs_fname.keys():
                for ob_var in Obs_fname[tmp_datstr].keys():
                    Obs_fname[tmp_datstr][ob_var] = np.sort(glob.glob(Obs_fname[tmp_datstr][ob_var]))



            
            if Obs_reloadmeth == 0:

                        
                for tmp_datstr in Obs_fname.keys():
                    Obs_dict[tmp_datstr] = {}
                    for ob_var in Obs_fname[tmp_datstr].keys():
                        print(datetime.now(),tmp_datstr,ob_var)
                        Obs_dict[tmp_datstr][ob_var] = {}
                        Obs_dict[tmp_datstr][ob_var]['Obs'] = []
                        Obs_dict[tmp_datstr][ob_var]['JULD'] = []
                        for tmpObsfname in Obs_fname[tmp_datstr][ob_var]:      

                            if ob_var in ['ProfT','ProfS']:
                                Obs_dict[tmp_datstr][ob_var]['Obs'].append(load_ops_prof_TS(tmpObsfname,ob_var[-1],excl_qc = True))
                            elif ob_var in ['SST_ins','SST_sat','SLA','ChlA']:
                                Obs_dict[tmp_datstr][ob_var]['Obs'].append(load_ops_2D_xarray(tmpObsfname,ob_var,excl_qc = False))

                            if ob_var in ['ProfT','ProfS','SST_ins','SST_sat','SLA','ChlA']:
                                tmptimetupel = datetime(*(Obs_dict[tmp_datstr][ob_var]['Obs'][-1]['JULD_datetime'][0]).timetuple()[:3])
                                Obs_dict[tmp_datstr][ob_var]['JULD'].append(  tmptimetupel  )

                            '''
                            if  ob_var == 'ChlA':
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_2D_xarray(tmpObsfname,'ChlA',excl_qc = False)
                                Obs_dict[tmp_datstr]['ChlA']['Obs'].append(tmp_Obs)  
                                if len(tmp_Obs['JULD_datetime'])>0:
                                    Obs_dict[tmp_datstr]['ChlA']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day)) 
                                else:
                                    Obs_dict[tmp_datstr]['ChlA']['JULD'].append(Obs_dict[tmp_datstr]['ChlA']['JULD'][-1] + timedelta(1)) 
                                del(tmp_Obs)
                            elif  ob_var == 'ProfT':
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_prof_TS(tmpObsfname,'T',excl_qc = True)
                                Obs_dict[tmp_datstr]['ProfT']['Obs'].append(tmp_Obs)  
                                Obs_dict[tmp_datstr]['ProfT']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day)) 
                            elif  ob_var =='ProfS' :
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_prof_TS(tmpObsfname,'S',excl_qc = True)
                                Obs_dict[tmp_datstr]['ProfS']['Obs'].append(tmp_Obs)  
                                Obs_dict[tmp_datstr]['ProfS']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day))
                                del(tmp_Obs)
                            elif  ob_var =='SST_ins' :
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_2D_xarray(tmpObsfname,'SST_ins',excl_qc = False)
                                Obs_dict[tmp_datstr]['SST_ins']['Obs'].append(tmp_Obs)  
                                Obs_dict[tmp_datstr]['SST_ins']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day))
                                del(tmp_Obs)
                            elif  ob_var =='SST_sat' :
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_2D_xarray(tmpObsfname,'SST_sat',excl_qc = False)
                                Obs_dict[tmp_datstr]['SST_sat']['Obs'].append(tmp_Obs)  
                                Obs_dict[tmp_datstr]['SST_sat']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day))
                                del(tmp_Obs)
                            elif  ob_var =='SLA' :
                                print(datetime.now(),tmpObsfname)
                                tmp_Obs = load_ops_2D_xarray(tmpObsfname,'SLA',excl_qc = False)
                                Obs_dict[tmp_datstr]['SLA']['Obs'].append(tmp_Obs)  
                                Obs_dict[tmp_datstr]['SLA']['JULD'].append(datetime(tmp_Obs['JULD_datetime'][0].year,tmp_Obs['JULD_datetime'][0].month,tmp_Obs['JULD_datetime'][0].day))
                                del(tmp_Obs)
                            '''
                        Obs_dict[tmp_datstr][ob_var]['JULD'] = np.array(Obs_dict[tmp_datstr][ob_var]['JULD'])
            
                
            elif Obs_reloadmeth > 0:
                for tmp_datstr in Obs_fname.keys():
                    Obs_dict[tmp_datstr] = {}
                    for ob_var in Obs_fname[tmp_datstr].keys():
                        print(datetime.now(),tmp_datstr,ob_var)
                        Obs_dict[tmp_datstr][ob_var] = {}
                        Obs_dict[tmp_datstr][ob_var]['Obs'] = []
                        Obs_dict[tmp_datstr][ob_var]['JULD'] = []

                        #for tmpObsfname in np.sort(glob.glob(Obs_fname[tmp_datstr][ob_var])):
                        for tmpObsfname in Obs_fname[tmp_datstr][ob_var]:
                            Obs_dict[tmp_datstr][ob_var]['Obs'].append({})  


                            rootgrp_obs = Dataset(tmpObsfname, 'r')
                            tmpObs_JULD =  rootgrp_obs.variables['JULD'][0:1].data[0]
                            tmpObs_JULD_REFERENCE = datetime.strptime(str(chartostring(rootgrp_obs.variables['JULD_REFERENCE'][:])),'%Y%m%d%H%M%S')
                            tmpObs_JULD_datetime = tmpObs_JULD_REFERENCE + timedelta(tmpObs_JULD)


                            Obs_dict[tmp_datstr][ob_var]['JULD'].append(datetime(tmpObs_JULD_datetime.year, tmpObs_JULD_datetime.month, tmpObs_JULD_datetime.day))   
                            rootgrp_obs.close()

                        Obs_dict[tmp_datstr][ob_var]['JULD'] = np.array(Obs_dict[tmp_datstr][ob_var]['JULD'])
        
                #pdb.set_trace()

            #pdb.set_trace()

    if do_memory:
        do_timer = True
 

    configlst = np.array([configd[ss] for ss in (configd)])
    uniqconfig = np.unique(configlst)



    # File name dictionary
    #==========================================
    # remove extra file names at the end of the list
    if trim_extra_files:
        fname_dict = remove_extra_end_file_dict(fname_dict)
    
    # Trim file list, using keywords [f0:f1:df] 
    if trim_files:
        fname_dict = trim_file_dict(fname_dict,thd)
    # create filesname dictionary
    Dataset_lst,nDataset = create_Dataset_lst(fname_dict)
    # create [empty] xarray handle dictionary
    xarr_dict = create_xarr_dict(fname_dict)


    # create colours and line styles for plots
    Dataset_col,Dataset_col_diff,linestyle_str = create_col_lst(nDataset)


    for tmp_datstr in Dataset_lst:
        if fig_lab_d[tmp_datstr] is None: fig_lab_d[tmp_datstr] = tmp_datstr


    axis_scale = 'Auto'

    if do_grad is None: do_grad = 0
    if do_cont is None: do_cont = True
    
    if verbose_debugging:
        print('======================================================')
        print('======================================================')
        print('=== Debugging printouts: verbose_debugging = True  ===')
        print('======================================================')
        print('======================================================')

    
    #Default variable for U and V flist
    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'


    z_meth_mat = ['z_slice','ss','nb','df','zm']

    #nav_lon_varname = 'nav_lon'
    #nav_lat_varname = 'nav_lat'
    #time_varname = 'time_counter'


    if use_cmocean:
        
        import cmocean
        # default color map to use
        base_cmap = cmocean.cm.thermal
        scnd_cmap = cmocean.cm.balance
        cylc_cmap = cmocean.cm.phase
    else:
        base_cmap = matplotlib.cm.viridis
        scnd_cmap = matplotlib.cm.coolwarm
        cylc_cmap = matplotlib.cm.hsv

    col_scl = 0
    curr_cmap = base_cmap
    #ncview style 4th order polynomial scaling. 
    base_cmap_low,base_cmap_high = scale_color_map(curr_cmap)

    #pdb.set_trace()

    if clim_sym is None: clim_sym = False
    if clim_sym == False:
        clim_sym_but = 0
    else:
        clim_sym_but = 1
    #clim_sym_but_norm_val = clim_sym

    # default initial indices
    if ii is None: ii = 10
    if jj is None: jj = 10
    if ti is None: ti = 0
    if zz is None: zz = 0
    if zz == 0: zi = 0
    if zi is None: zi = 0
    #pdb.set_trace()

    if fig_dir is None:

        #fig_dir = os.getcwd() + '/tmpfigs'
        #fig_dir = script_dir + './tmpfigs'
        fig_dir = './tmpfigs'
        print('fig_dir: ',fig_dir )

    #need to load lon_mat and lat_mat to implement lon_in and lat_in
    #need to load date_mat to implement date_in_ind

    # Set mode: Click or Loop
    mode = 'Click'
    loop_sleep = 0.01


    # if a secondary data set, give ability to change data sets. 
    #secdataset_proc_list = ['Dataset 1', 'Dataset 2', 'Dat2-Dat1', 'Dat1-Dat2']

    secdataset_proc_list = Dataset_lst.copy()
    if nDataset > 1:
        '''
        for tmp_datstr in Dataset_lst[1:]:
            th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
            secdataset_proc_list.append('Dat%i-Dat1'%th_d_ind)
            secdataset_proc_list.append('Dat1-Dat%i'%th_d_ind)

        '''
        Dataset_col_diff_dict = {}
        cnt_diff_str_name = 0
        for tmp_datstr1 in Dataset_lst:
            #th_d_ind1 = int(tmp_datstr1[-1])
            th_d_ind1 = int(tmp_datstr1[8:])
            for tmp_datstr2 in Dataset_lst:
                #th_d_ind2 = int(tmp_datstr2[-1])
                th_d_ind2 = int(tmp_datstr2[8:])
                if tmp_datstr1!=tmp_datstr2:
                #if ((nDataset<6)&(th_d_ind1 != th_d_ind2))|((nDataset>=6)&(th_d_ind1 > th_d_ind2)):
                    tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)
                    secdataset_proc_list.append(tmp_diff_str_name)
                    
                    Dataset_col_diff_dict[tmp_diff_str_name] = Dataset_col_diff[cnt_diff_str_name]
                    cnt_diff_str_name = cnt_diff_str_name+1

    if secdataset_proc is None: secdataset_proc = Dataset_lst[0]

    if load_second_files == False:
        clim_pair = False
    
    if justplot is None: justplot = False


    if hov_time is None: hov_time = True

    print('thin: %i; thin_files: %i; hov_time: %s; '%(thd[1]['dx'],thd[1]['df'],hov_time))
    pxy = thd[1]['pxy']
    print('maximum pixels plotted in x and y dir pxy:',pxy)

    nlon_amm7 = 297
    nlat_amm7 = 375
    nlon_amm15 = 1458
    nlat_amm15 = 1345


    # create a dictionary with all the config info in it
    config_fnames_dict = create_config_fnames_dict(configd,Dataset_lst,script_dir)

    init_timer.append((datetime.now(),'Indices set'))
    
    z_meth_default = config_fnames_dict[configd[1]]['z_meth_default']
    
    #set ncvariable names for mesh files
    ncgdept,nce1t,nce2t,nce3t,ncglamt,ncgphit = create_gdept_ncvarnames(config_fnames_dict,configd)
   
    init_timer.append((datetime.now(),'Config files read'))

    # create dictionary with mesh files handles
    rootgrp_gdept_dict = create_rootgrp_gdept_dict(config_fnames_dict,Dataset_lst,configd, use_xarray_gdept = use_xarray_gdept)
    
    init_timer.append((datetime.now(),'Gdept opened'))

    #pdb.set_trace()

    """
    #config version specific info - mainly grid, and lat/lon info
    if configd[1].upper() == 'AMM7':
        #grid lat lon
        lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
        lat = np.arange(40.066669,65+1/15.,1/15.)

    elif configd[1].upper() == 'GULF18':

        #grid lat lon
        lon = rootgrp_gdept_dict['Dataset 1'].variables[ncglamt][:,0,:].ravel()
        lat = rootgrp_gdept_dict['Dataset 1'].variables[ncgphit][:,:,0].ravel()
    """
    if z_meth is None:
        z_meth = z_meth_default
    '''

    #default regirdding settings
    regrid_meth = 1 # NN
    regrid_params = None
    regrid_params = {}  
    regrid_params['do_regrid'] = 0


    # regridding indices.
    for tmp_datstr in Dataset_lst[1:]:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        #rootgrp_gdept_dict[tmp_datstr] = rootgrp_gdept_dict['Dataset 1']
        

        regrid_params[tmp_datstr] = None#(None,None,None,None,None)
        if (configd[th_d_ind] is not None) & (configd[th_d_ind]!=configd[1]):
        #if (configd[th_d_ind] is not None) :

            if (configd[1].upper() in ['AMM7','AMM15']) & (configd[th_d_ind].upper() in ['AMM7','AMM15']):  
                #mesh_file_2nd = config_fnames_dict[configd[th_d_ind]]['mesh_file'] 
                #rootgrp_gdept_dict[tmp_datstr] = Dataset(mesh_file_2nd, 'r', format='NETCDF4')

                regrid_params['do_regrid'] = 2

                if (configd[1].upper() == 'AMM15') & (configd[th_d_ind].upper() == 'AMM7'):  


                    lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
                    lat = np.arange(40.066669,65+1/15.,1/15.)


                    amm_conv_dict = {}
                    rootgrp = Dataset(config_fnames_dict[configd[1]]['regrid_amm7_amm15'], 'r')
                    for var_conv in rootgrp.variables.keys(): amm_conv_dict[var_conv] = rootgrp.variables[var_conv][:]
                    rootgrp.close()
        
                    nlon_amm        = nlon_amm15
                    nlat_amm        = nlat_amm15
                    nlon_amm_2nd    = nlon_amm7
                    nlat_amm_2nd    = nlat_amm7


                elif (configd[1].upper() == 'AMM7') & (configd[th_d_ind].upper() == 'AMM15'):

                    amm_conv_dict = {}
                    rootgrp = Dataset(config_fnames_dict[configd[th_d_ind]]['regrid_amm15_amm7'], 'r')
                    for var_conv in rootgrp.variables.keys(): amm_conv_dict[var_conv] = rootgrp.variables[var_conv][:]
                    rootgrp.close()
        
                    nlon_amm        = nlon_amm7
                    nlat_amm        = nlat_amm7
                    nlon_amm_2nd    = nlon_amm15
                    nlat_amm_2nd    = nlat_amm15

                regrid_params[tmp_datstr] = regrid_2nd_thin_params(amm_conv_dict,nlon_amm,nlat_amm, nlon_amm_2nd,nlat_amm_2nd,thd)
                #pdb.set_trace()
            else:

                #NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out
                print('Differing configs not only amm15 and amm7')
                pdb.set_trace()
    '''
    init_timer.append((datetime.now(),'config 2 params'))


    init_timer.append((datetime.now(),'xarray open_mfdataset connecting'))
    print('xarray open_mfdataset, Start',datetime.now())

    #pdb.set_trace()
    if ld_lst is None:
        nldi = 0
        ldi_ind_mat, ld_lab_mat = None, None
    else:
        if isinstance(ld_lst, str):
            ldi_ind_mat = np.array([int(ss) for ss in ld_lst.split(',')])
            nldi = ldi_ind_mat.size
        else:
            pdb.set_trace()

        if ld_lab_lst is None:
            ld_lab_mat = np.array(['%i'%(ii) for ii in ldi_ind_mat])
        else:
            ld_lab_mat = np.array(ld_lab_lst.split(','))

    # connect to files with xarray, and create dictionaries with vars, dims, grids, time etc. S
    #pdb.set_trace()
    var_d,var_dim,var_grid,ncvar_d,ncdim_d,time_d  = connect_to_files_with_xarray(Dataset_lst,fname_dict,xarr_dict,nldi,ldi_ind_mat, ld_lab_mat,ld_nctvar,force_dim_d = force_dim_d,xarr_rename_master_dict=xarr_rename_master_dict)
    

    # tmp = xarr_dict['Dataset 1']['T'][0].groupby('time_counter.year').groupby('time_counter.month').mean('time_counter') 
    
    #nctime = xarr_dict['Dataset 1']['T'][0].variables['time_counter']
    #xarr_dict['Dataset 1']['T'][0] = xarr_dict['Dataset 1']['T'][0].groupby('time_counter.month').mean('time_counter') 
    
    # resample to give monthly means etc. 

    init_timer.append((datetime.now(),'xarray open_mfdataset connected'))

    #pdb.set_trace()
    # get lon, lat and time names from files
    nav_lon_varname_dict,nav_lat_varname_dict,time_varname_dict,nav_lon_var_mat,nav_lat_var_mat,time_varname_mat = create_ncvar_lon_lat_time_dict(ncvar_d)
    #time_varname = time_varname_dict['Dataset 1']
    
    init_timer.append((datetime.now(),'created ncvar lon lat time'))

    if resample_freq is not None:
        #pdb.set_trace()
        print('xarray open_mfdataset: Start resample with %s'%(resample_freq), datetime.now())
        xarr_dict = resample_xarray(xarr_dict,resample_freq,time_varname_dict)
        print('xarray open_mfdataset: Finish resample with %s'%(resample_freq), datetime.now())
        init_timer.append((datetime.now(),'xarray resampled'))

    #xarr_dict['Dataset 1']['T'][0] = xarr_dict['Dataset 1']['T'][0].resample(time_counter = '1m').mean()

    #tmp = xarr_dict['Dataset 1']['T'][0].groupby('time_counter.year').groupby('time_counter.month').mean('time_counter') 
    #pdb.set_trace()
   


            

    print ('xarray open_mfdataset Finish',datetime.now())


    # Create lon and lat dictionaries
    lon_d,lat_d = create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutxind,cutyind,cutout_data)
    

    domsize = {}
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:])
        domsize[th_d_ind] = np.array(lat_d[th_d_ind].shape)
    #pdb.set_trace()

    #pdb.set_trace()
    xypos_dict = {}
    
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        tmpconfig = configd[th_d_ind]
        xypos_dict[tmp_datstr] = {}
        xypos_dict[tmp_datstr]['do_xypos'] = False
        
        if 'xypos_file' in config_fnames_dict[tmpconfig].keys():

            xypos_dict[tmp_datstr]['do_xypos'] = True
            rootgrp = Dataset(config_fnames_dict[tmpconfig]['xypos_file'], 'r')
            for xy_var in rootgrp.variables.keys(): xypos_dict[tmp_datstr][xy_var] = rootgrp.variables[xy_var][:]
            xypos_dict[tmp_datstr]['lon_min'] = xypos_dict[tmp_datstr]['LON'].min()
            xypos_dict[tmp_datstr]['lat_min'] = xypos_dict[tmp_datstr]['LAT'].min()
            xypos_dict[tmp_datstr]['dlon'] =  (np.diff(xypos_dict[tmp_datstr]['LON'][0,:])).mean()
            xypos_dict[tmp_datstr]['dlat'] =  (np.diff(xypos_dict[tmp_datstr]['LAT'][:,0])).mean()
            
            rootgrp.close()



            nxylat, nxylon = xypos_dict[tmp_datstr]['LAT'].shape
            xypos_mask =  np.ma.getmaskarray(xypos_dict[tmp_datstr]['XPOS'])

            xypos_xmat, xypos_ymat = np.meshgrid(np.arange(nxylon), np.arange(nxylat))

            points = (xypos_xmat[~xypos_mask], xypos_ymat[~xypos_mask])
            values_X = xypos_dict[tmp_datstr]['XPOS'][~xypos_mask]
            values_Y = xypos_dict[tmp_datstr]['YPOS'][~xypos_mask]

            #plt.plot(points[0],points[1],'x')
            #plt.show()
            #pdb.set_trace()
            
            xypos_dict[tmp_datstr]['XPOS_NN'] = griddata(points, values_X, (xypos_xmat, xypos_ymat), method='nearest')
            xypos_dict[tmp_datstr]['YPOS_NN'] = griddata(points, values_Y, (xypos_xmat, xypos_ymat), method='nearest')

            
            '''
            
            ntest_i, ntest_j = lon_d[th_d_ind].shape
            test_xpos_ii, test_xpos_jj = np.arange(0,ntest_i-1,10),np.arange(0,ntest_j-1,10)
            res_xpos_ii_mat, res_xpos_jj_mat = ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,{},lon_d[th_d_ind][0:ntest_i-1:10,0:ntest_j-1:10], lat_d[th_d_ind][0:ntest_i-1:10,0:ntest_j-1:10])
            res_xpos_ii = res_xpos_ii_mat.mean(axis = 1)
            res_xpos_jj = res_xpos_jj_mat.mean(axis = 0)

            test_xpos_ii_rmse = np.sqrt(((test_xpos_ii-res_xpos_ii)**2).mean())
            test_xpos_jj_rmse = np.sqrt(((test_xpos_jj-res_xpos_jj)**2).mean())

            if (test_xpos_ii_rmse>2)| (test_xpos_jj_rmse>2):

                print('check correct xypos file. test rmse >2',test_xpos_ii_rmse, test_xpos_jj_rmse)

                pdb.set_trace()
                plt.subplot(2,2,1)
                plt.plot(test_xpos_ii,res_xpos_ii)
                plt.subplot(2,2,2)
                plt.plot(test_xpos_jj,res_xpos_jj)
                plt.subplot(2,2,3)
                plt.plot(test_xpos_ii-res_xpos_ii)
                plt.subplot(2,2,4)
                plt.plot(test_xpos_jj-res_xpos_jj)
                plt.show()


                pdb.set_trace()


            '''


    init_timer.append((datetime.now(),'created lon lat dict'))
    # if use key words to set intial lon/lat,nvarbutcol convert to jj/ii
    if (lon_in is not None) & (lat_in is not None):

        lonlatin_dist_mat = np.sqrt((lon_d[1] - lon_in)**2 + (lat_d[1] - lat_in)**2)
        #jj,ii = lonlatin_dist_mat.argmin()//lon_d[1].shape[1], lonlatin_dist_mat.argmin()%lon_d[1].shape[1]
        jj,ii = lonlatin_dist_mat.argmin()//domsize[1][1], lonlatin_dist_mat.argmin()%domsize[1][1]


    init_timer.append((datetime.now(),'Lon/Lats loaded'))
    do_mask_dict = {}
    for tmp_datstr in Dataset_lst:
        tmp_do_mask = do_mask
        if tmp_do_mask:
            if 'tmask' not in rootgrp_gdept_dict[tmp_datstr].variables.keys():
                
                tmp_do_mask = False
        do_mask_dict[tmp_datstr] = tmp_do_mask
    #pdb.set_trace()       
    #create depth (gdept) dictionary
    grid_dict,nz = load_grid_dict(Dataset_lst,rootgrp_gdept_dict, thd, nce1t,nce2t,nce3t,configd, config_fnames_dict,cutxind,cutyind,cutout_data, do_mask_dict)
    #pdb.set_trace()
    # if using WW3 grid, load regridding interpolation weights


     
    init_timer.append((datetime.now(),'grid_dict, nz loaded'))

    if 'WW3' in ncdim_d['Dataset 1']:
         
        grid_dict['WW3'] = {}
        tmpfname_out_WW3_amm15_bilin = '/data/cr1/hadjt/data/reffiles/SSF/regrid_WW3_amm15_nn_mask.nc'
        tmpfname_out_WW3_amm15_bilin = '/data/users/jonathan.tinker/reffiles/SSF/regrid_WW3_amm15_nn_mask.nc'
        rootgrp = Dataset(tmpfname_out_WW3_amm15_bilin, 'r', format='NETCDF4')
        grid_dict['WW3']['NWS_WW3_nn_ind'] = rootgrp.variables['NWS_WW3_nn_ind'][:,:]
        grid_dict['WW3']['AMM15_mask'] = rootgrp.variables['AMM15_mask'][:,:].astype('bool')
        rootgrp.close()  
        init_timer.append((datetime.now(),'WW3 added to grid_dict'))
     

    #pdb.set_trace()
    if var is None: 
        if 'votemper' in var_d[1]['mat']:
            var = 'votemper'
        else:
            var = var_d[1]['mat'][0]

    if var not in var_d[1]['mat']: var = var_d[1]['mat'][0]

    nice_varname_dict = {}
    for tmpvar in var_d[1]['mat']: nice_varname_dict[tmpvar] = tmpvar

    nice_varname_dict['thetao'] = 'Temperature'
    nice_varname_dict['thetao'] = 'Temperature'
    nice_varname_dict['votemper'] = 'Temperature'
    nice_varname_dict['vosaline'] = 'Salinity'
    nice_varname_dict['pea'] = 'Potential Energy Anomaly'
    nice_varname_dict['peat'] = 'Potential Energy Anomaly (T component)'
    nice_varname_dict['peas'] = 'Potential Energy Anomaly (S component)'
    nice_varname_dict['rho'] = 'Density'
    nice_varname_dict['N2'] = 'Brunt-Väisäla frequency'
    nice_varname_dict['N2max'] = 'Brunt-Väisäla frequency - depth of max'
    nice_varname_dict['Pync_Z'] = 'Depth of pycnocline'
    nice_varname_dict['Pync_Th'] = 'Thickness of pycnocline '

    nice_varname_dict['baroc_mag'] = 'Baroclinic current magnitude'
    nice_varname_dict['barot_mag'] = 'Barotropic current magnitude'
    nice_varname_dict['baroc_phi'] = 'Baroclinic current phase (degrees)'
    nice_varname_dict['barot_phi'] = 'Barotropic current phase (degrees)'

    nice_varname_dict['baroc_curl'] = 'Baroclinic current curl'
    nice_varname_dict['barot_curl'] = 'Barotropic current curl'

    nice_varname_dict['baroc_div'] = 'Baroclinic current divergence'
    nice_varname_dict['barot_div'] = 'Barotropic current divergence'

    nice_varname_dict['vozocrtx'] = 'Baroclinic current (eastward component)'
    nice_varname_dict['vomecrty'] = 'Baroclinic current (westward component)'

    nice_varname_dict['sossheig'] = 'Sea surface height'
    nice_varname_dict['temper_bot'] = 'Bottom temperature'
    nice_varname_dict['tempis_bot'] = 'Bottom (in situ) temperature'
    nice_varname_dict['votempis'] = 'Temperature (in situ)'
    nice_varname_dict['mld25h_1'] = 'Mixed layer depth (version 1)'
    nice_varname_dict['mld25h_2'] = 'Mixed layer depth (version 2)'
    nice_varname_dict['karamld'] = 'Mixed layer depth (Kara)'
    nice_varname_dict['somxl010'] = 'Mixed Layer Depth (dsigma = 0.01 wrt 10m)'
    nice_varname_dict['somxzint'] = 'Mixed Layer Depth interpolated'
    nice_varname_dict['sokaraml'] = 'Mixed Layer Depth interpolated'

    # ERSEM
    nice_varname_dict['pCO2'] = 'Carbonate pCO2'
    nice_varname_dict['CHL'] = 'Total Chlorophyll'
    nice_varname_dict['netPP'] = 'Net Primary Production'
    nice_varname_dict['N1p'] = 'Phosphate'
    nice_varname_dict['N3n'] = 'Nitrate'
    nice_varname_dict['N5s'] = 'Silicate'
    nice_varname_dict['N4n'] = 'Ammonium Nitrogen'
    nice_varname_dict['O2o'] = 'Oxygen'

    nice_varname_dict['N:P'] = 'Nitrate to Phosphate Ratio'


    nice_varname_dict['pH'] = 'Carbonate pH'
    nice_varname_dict['PhytoC'] = 'Phytoplankton (carbon)'
    nice_varname_dict['Visib'] = 'Secchi depth '
    nice_varname_dict['spCO2'] = 'Surface Carbonate pCO2'

    # Increment values
    nice_varname_dict['wnd_mag'] = 'Wind speed'
    nice_varname_dict['bckint'] = 'Temperature Increment'
    nice_varname_dict['bckins'] = 'Salinity Increment'
    nice_varname_dict['bckinu'] = 'Velocity (eastward component) Increment'
    nice_varname_dict['bckinv'] = 'Velocity (northward component) Increment'
    nice_varname_dict['bckineta'] = 'Sea-surface height Increment'
    nice_varname_dict['bckinSST_28_OBIAS'] = 'SST (#28) Increment'
    nice_varname_dict['bckinSST_24_OBIAS'] = 'SST (#24) Increment'
    nice_varname_dict['bckinSST_44_OBIAS'] = 'SST (#44) Increment'
    nice_varname_dict['bckinSST_38_OBIAS'] = 'SST (#38) Increment'
    nice_varname_dict['bckinSLA_ALL1_OBIAS'] = 'SLA (All) Increment'

    # CICE Sea Ice Variables
    nice_varname_dict['hs'] = 'grid cell mean snow thickness'
    nice_varname_dict['Tsfc'] = 'snow/ice surface temperature'
    nice_varname_dict['aice'] = 'ice area  (aggregate)'
    nice_varname_dict['uvel'] = 'ice velocity (x)'
    nice_varname_dict['vvel'] = 'ice velocity (y)'
    nice_varname_dict['snow_ai'] = 'snowfall rate'
    nice_varname_dict['fswabs_ai'] = 'snow/ice/ocn absorbed solar flux'
    nice_varname_dict['alvdr'] = 'visible direct albedo'
    nice_varname_dict['alidr'] = 'near IR direct albedo'
    nice_varname_dict['flat_ai'] = 'latent heat flux'
    nice_varname_dict['fsens_ai'] = 'sensible heat flux'
    nice_varname_dict['flwup_ai'] = 'upward longwave flux'
    nice_varname_dict['evap_ai'] = 'evaporative water flux'
    nice_varname_dict['congel'] = 'congelation ice growth'
    nice_varname_dict['frazil'] = 'frazil ice growth'
    nice_varname_dict['snoice'] = 'snow-ice formation'
    nice_varname_dict['meltt'] = 'top ice melt'
    nice_varname_dict['melts'] = 'top snow melt'
    nice_varname_dict['meltb'] = 'basal ice melt'
    nice_varname_dict['meltl'] = 'lateral ice melt'
    nice_varname_dict['strairx'] = 'atm/ice stress (x)'
    nice_varname_dict['strairy'] = 'atm/ice stress (y)'
    nice_varname_dict['shear'] = 'strain rate (shear)'
    nice_varname_dict['dvidtt'] = 'ice volume tendency thermo'
    nice_varname_dict['dvsdtt'] = 'snow volume tendency thermo'
    nice_varname_dict['dvidtd'] = 'ice volume tendency dynamics'
    nice_varname_dict['dvsdtd'] = 'snow volume tendency dynamics'
    nice_varname_dict['dvidti'] = 'ice volume tendency assimilation'
    nice_varname_dict['dvsdti'] = 'snow volume tendency assimilation'
    nice_varname_dict['dardg1dt'] = 'ice area ridging rate'
    nice_varname_dict['dardg2dt'] = 'ridge area formation rate'
    nice_varname_dict['dvirdgdt'] = 'ice volume ridging rate'
    nice_varname_dict['opening'] = 'lead area opening rate'
    nice_varname_dict['apond'] = 'melt pond fraction of sea ice'
    nice_varname_dict['apond_ai'] = 'melt pond fraction of grid cell'
    nice_varname_dict['hpond'] = 'mean melt pond depth over sea ice'
    nice_varname_dict['hpond_ai'] = 'mean melt pond depth over grid cell'
    nice_varname_dict['ipond'] = 'mean pond ice thickness over sea ice'
    nice_varname_dict['ipond_ai'] = 'mean pond ice thickness over grid cell'
    nice_varname_dict['apeff'] = 'radiation-effective pond area fraction of sea ice'
    nice_varname_dict['apeff_ai'] = 'radiation-effective pond area fraction over grid cell'
    nice_varname_dict['aicen'] = 'ice area, categories'
    nice_varname_dict['vicen'] = 'ice volume, categories'
    nice_varname_dict['vsnon'] = 'snow depth on ice, categories'
    nice_varname_dict['fsurfn_ai'] = 'net surface heat flux, categories'
    nice_varname_dict['fcondtopn_ai'] = 'top sfc conductive heat flux, cat'
    nice_varname_dict['apondn'] = 'melt pond fraction, category'
    nice_varname_dict['hpondn'] = 'melt pond depth, category'
    nice_varname_dict['apeffn'] = 'effective melt pond fraction, category'
    nice_varname_dict['Sinz'] = 'ice internal bulk salinity'


    '''
    nice_varname_dict['VolTran_e3_mag'] = 'VolTran_e3_mag'
    nice_varname_dict['VolTran_e3_div'] = 'VolTran_e3_div'
    nice_varname_dict['VolTran_e3_curl'] = 'VolTran_e3_curl'

    
    nice_varname_dict['VolTran_mag'] = 'VolTran_mag'
    nice_varname_dict['VolTran_div'] = 'VolTran_div'
    nice_varname_dict['VolTran_curl'] = 'VolTran_curl'
    nice_varname_dict['StreamFunction'] = 'StreamFunct'
    nice_varname_dict['StreamFunction_e3'] = 'StreamFunct_e3'
    '''

    #WW3 var names:
    #nice_varname_dict['cx'] = 'longitude cell size factor'
    #nice_varname_dict['cy'] = 'latitude cell size factor'
    #nice_varname_dict['standard_longitude'] = 'longitude'
    #nice_varname_dict['standard_latitude'] = 'latitude'
    #nice_varname_dict['time'] = 'time'
    #nice_varname_dict['forecast_period'] = 'forecast period'
    #nice_varname_dict['forecast_reference_time'] = 'forecast reference time'
    nice_varname_dict['dpt'] = 'depth'
    nice_varname_dict['ucur'] = 'eastward current'
    nice_varname_dict['vcur'] = 'northward current'
    nice_varname_dict['uwnd'] = 'eastward_wind'
    nice_varname_dict['vwnd'] = 'northward_wind'
    nice_varname_dict['hs'] = 'significant height of wind and swell waves'
    nice_varname_dict['t02'] = 'mean period T02'
    nice_varname_dict['t0m1'] = 'mean period T0m1'
    nice_varname_dict['t01'] = 'mean period T01'
    nice_varname_dict['dir'] = 'wave mean direction'
    nice_varname_dict['spr'] = 'directional spread'
    nice_varname_dict['dp'] = 'peak direction'
    nice_varname_dict['tp'] = 'wave peak period'
    nice_varname_dict['phs0'] = 'wave significant height partition 0'
    nice_varname_dict['phs1'] = 'wave significant height partition 1'
    nice_varname_dict['phs2'] = 'wave significant height partition 2'
    nice_varname_dict['phs3'] = 'wave significant height partition 3'
    nice_varname_dict['ptp0'] = 'peak period partition 0'
    nice_varname_dict['ptp1'] = 'peak period partition 1'
    nice_varname_dict['ptp2'] = 'peak period partition 2'
    nice_varname_dict['ptp3'] = 'peak period partition 3'
    nice_varname_dict['pdir0'] = 'wave mean direction partition 0'
    nice_varname_dict['pdir1'] = 'wave mean direction partition 1'
    nice_varname_dict['pdir2'] = 'wave mean direction partition 2'
    nice_varname_dict['pdir3'] = 'wave mean direction partition 3'
    nice_varname_dict['pspr0'] = 'directional spread partition 0'
    nice_varname_dict['pspr1'] = 'directional spread partition 1'
    nice_varname_dict['pspr2'] = 'directional spread partition 2'
    nice_varname_dict['pspr3'] = 'directional spread partition 3'
    nice_varname_dict['pt01c0'] = 'mean period T01 partition 0'
    nice_varname_dict['pt01c1'] = 'mean period T01 partition 1'
    nice_varname_dict['pt01c2'] = 'mean period T01 partition 2'
    nice_varname_dict['pt01c3'] = 'mean period T01 partition 3'
    nice_varname_dict['pt02c0'] = 'mean period T02 partition 0'
    nice_varname_dict['pt02c1'] = 'mean period T02 partition 1'
    nice_varname_dict['pt02c2'] = 'mean period T02 partition 2'
    nice_varname_dict['pt02c3'] = 'mean period T02 partition 3'
    nice_varname_dict['pep0'] = 'energy at peak frequency partition 0'
    nice_varname_dict['pep1'] = 'energy at peak frequency partition 1'
    nice_varname_dict['pep2'] = 'energy at peak frequency partition 2'
    nice_varname_dict['pep3'] = 'energy at peak frequency partition 3'
    nice_varname_dict['uust'] = 'eastward friction velocity'
    nice_varname_dict['vust'] = 'northward friction velocity'
    nice_varname_dict['cha'] = 'charnock coefficient for surface roughness length for momentum in air'
    nice_varname_dict['utaw'] = 'eastward wave supported wind stress'
    nice_varname_dict['vtaw'] = 'northward wave supported wind stress'
    nice_varname_dict['utwo'] = 'eastward wave to ocean stress'
    nice_varname_dict['vtwo'] = 'northward wave to ocean stress'
    nice_varname_dict['uuss'] = 'eastward surface stokes drift'
    nice_varname_dict['vuss'] = 'northward surface stokes drift'


    
    init_timer.append((datetime.now(),'Nice names loaded'))

    # extract time information from xarray.
    # needs to work for gregorian and 360 day calendars.
    # needs to work for as x values in a plot, or pcolormesh
    # needs work, xarray time is tricky

    init_timer.append((datetime.now(),'nc time started'))

    #pdb.set_trace()
    time_datetime,time_datetime_since_1970,ntime,ti, nctime_calendar_type = extract_time_from_xarr(xarr_dict['Dataset 1']['T'],fname_dict['Dataset 1']['T'][0],time_varname_dict['Dataset 1'],ncdim_d['Dataset 1']['T']['t'],date_in_ind,date_fmt,ti,verbose_debugging)

    init_timer.append((datetime.now(),'nc time completed'))


    if justplot: 
        print('justplot:',justplot)
        print('Just plotting, and exiting, not interactive.')
        
        just_plt_cnt = 0
        njust_plt_cnt = 0

        if (justplot_date_ind is None)|(justplot_date_ind == 'None'):
             justplot_date_ind = time_datetime[ti].strftime(date_fmt)

        if (justplot_z_meth_zz is None)|(justplot_z_meth_zz == 'None'):
             justplot_z_meth_zz = 'ss:0,nb:0,df:0'

        if (justplot_secdataset_proc is None)|(justplot_secdataset_proc == 'None'):
             justplot_secdataset_proc = 'Dataset_1,Dataset_2,Dat2-Dat1'

        justplot_secdataset_proc = justplot_secdataset_proc.replace('_',' ')

        
        justplot_date_ind_lst = justplot_date_ind.split(',')
        justplot_z_meth_zz_lst = justplot_z_meth_zz.split(',')
        justplot_secdataset_proc_lst = justplot_secdataset_proc.split(',')
                
        
        just_plt_vals = []
        # cycle though dates. Change ti, so data_inst reloaded automatically.
        #     location unchanged, so hov doen't need reloading, but everything else does
        for jpdti,justplot_date_ind_str in enumerate(justplot_date_ind_lst):
            # cycle through depths. Map, and ts need changing, but profile hov and cross sectons, don't
            for jpzmi, justplot_z_meth_zz in enumerate(justplot_z_meth_zz_lst): 
                #pdb.set_trace()
                justplot_z_meth,justplot_zz_str = justplot_z_meth_zz.split(':')
                justplot_zz = int(justplot_zz_str)
                #  cycle through datasets, nothing needs reloading. 
                for jpspi, secdataset_proc in enumerate(justplot_secdataset_proc_lst): 
                    just_plt_vals.append((secdataset_proc,justplot_date_ind_str, justplot_z_meth,justplot_zz, True, True, True, False, True))
                    njust_plt_cnt+=1
                    '''
                    if (jpspi == 0):#                                                                     reload_map,reload_ew,reload_ns,reload_hov,reload_ts,                        
                        just_plt_vals.append((secdataset_proc,justplot_date_ind_str, justplot_z_meth,justplot_zz, True, True, True, False, False))
                    else:
                        just_plt_vals.append((secdataset_proc,justplot_date_ind_str, justplot_z_meth,justplot_zz, False, False, False, False, False))
                    '''  
    init_timer.append((datetime.now(),'justplot prepared'))
    # repeat if comparing two time series. 
    if load_second_files:
        
        #var_d[2] = {}
        clim_sym = True
        

        init_timer.append((datetime.now(),'nc time 2nd started'))


        # load time from second data set to check if matches with first dataset.         

        time_datetime_2nd,time_datetime_since_1970_2nd,ntime_2nd,ti, nctime_calendar_type = extract_time_from_xarr(xarr_dict['Dataset 2']['T'],fname_dict['Dataset 2']['T'][0],
            time_varname_dict['Dataset 2'],ncdim_d['Dataset 2']['T']['t'],date_in_ind,date_fmt,ti,verbose_debugging)


        
        # check both datasets have the same times
        if allow_diff_time == False:
            if ntime_2nd != ntime:     
                print()
                print('Diff Times have different number of files. To Continue press c')
                print()
                pdb.set_trace() 
            else:
                if (time_datetime_since_1970_2nd != time_datetime_since_1970).any():   
                    print()
                    print("Times don't match between Dataset 1 and Dataset 2. To Continue press c")
                    print()
                    pdb.set_trace()



        init_timer.append((datetime.now(),'nc time 2nd completed'))

        # check both datasets have the same lons and lats (if same config)
        #if configd[2] is None:
        #if configd[2] != configd[1]:
        #for tmp_datstr in Dataset_lst[1:]:
        #    th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

        if configd[2] == configd[1]:
            #pdb.set_trace()
            #if (np.array(lat_d[1].shape)==np.array(lat_d[2].shape)).all():
            if (np.array(domsize[1])==np.array(domsize[2])).all():
                if (lat_d[1] != lat_d[2]).any():
                    print('Diff nav_lat_2nd dont match')
                    pdb.set_trace()
                if (lon_d[1] != lon_d[2]).any():
                    print('Diff nav_lon_2nd dont match')
                    pdb.set_trace()
            else:
                print('Diff nav_lat_2nd shape doesnt match')
                #pdb.set_trace()

        # use a difference colormap if comparing files
        curr_cmap = scnd_cmap

    
    # open file list with xarray
    for tmp_datstr in Dataset_lst: # xarr_dict.keys():
        #time_d[tmp_datstr] = {}
        for tmpgrid in xarr_dict[tmp_datstr].keys():
            '''
            for ncvar in ncvar_d[tmp_datstr][tmpgrid]: 
                #if ncvar.upper() in nav_lon_var_mat: nav_lon_varname = ncvar
                #if ncvar.upper() in nav_lat_var_mat: nav_lat_varname = ncvar
                if ncvar.upper() in time_varname_mat: tmp_time_varname = ncvar

            #if ncvar.upper() in time_varname_mat: time_varname = ncvar

            #time_d[tmp_datstr][tmpgrid] = {}
            '''
            #pdb.set_trace()    
            #if tmpgrid == 'I': continue #  no longer need to skip for Increments, as extract_time_from_xarr handles it
            (time_d[tmp_datstr][tmpgrid]['datetime'],time_d[tmp_datstr][tmpgrid]['datetime_since_1970'],tmp_ntime,tmp_ti, nctime_calendar_type) =  extract_time_from_xarr(xarr_dict[tmp_datstr][tmpgrid],fname_dict[tmp_datstr][tmpgrid][0], time_varname_dict[tmp_datstr],ncdim_d[tmp_datstr][tmpgrid]['t'],date_in_ind,date_fmt,ti,verbose_debugging)

    # add derived variables
    var_d,var_dim, var_grid = add_derived_vars(var_d,var_dim, var_grid, Dataset_lst)

    # add derived variales to nice names if mising. 

    for ss in var_d['d']: 
        if ss not in nice_varname_dict.keys():
            nice_varname_dict[ss] = ss


    add_TSProf = False
    if ('votemper' in var_d[1]['mat']) & ('vosaline' in var_d[1]['mat']):
        add_TSProf = True

    if (tmp_var_U in var_d[1]['mat']) & (tmp_var_V in var_d[1]['mat']):
        if vis_curr == -1:
            vis_curr = 0


    #pdb.set_trace()

    ldi = 0 

    data_inst = None
    data_inst_mld = None
    data_inst_U = None
    if preload_data:
        preload_data_ti = ti
        preload_data_ti_mld = ti
        preload_data_ti_U = ti

        preload_data_var = var

        preload_data_ldi = ldi
        preload_data_ldi_mld = ldi
        preload_data_ldi_U = ldi
    
    init_timer.append((datetime.now(),'Derived var defined'))

    rot_dict = {}
    """
    if (configd[1].upper() in ['AMM15','CO9P2']): 
        lon_rotamm15,lat_rotamm15 = reduce_rotamm15_grid(lon_d['amm15'], lat_d['amm15'])

        dlon_rotamm15 = (np.diff(lon_rotamm15)).mean()
        dlat_rotamm15 = (np.diff(lat_rotamm15)).mean()
        nlon_rotamm15 = lon_rotamm15.size
        nlat_rotamm15 = lat_rotamm15.size

        
        rot_dict[configd[1]] = {}
        rot_dict[configd[1]]['dlon'] = dlon_rotamm15
        rot_dict[configd[1]]['dlat'] = dlat_rotamm15
        rot_dict[configd[1]]['nlon'] = nlon_rotamm15
        rot_dict[configd[1]]['nlat'] = nlat_rotamm15
        rot_dict[configd[1]]['lon_rot'] = lon_rotamm15
        rot_dict[configd[1]]['lat_rot'] = lat_rotamm15   
        
        del(dlon_rotamm15)
        del(dlat_rotamm15)

        del(nlon_rotamm15)
        del(nlat_rotamm15)
        del(lon_rotamm15)
        del(lat_rotamm15)


    if load_second_files:
        #if configd[2] is not None:
        if (configd[2].upper() in ['AMM15','CO9P2']):
            lon_rotamm15,lat_rotamm15 = reduce_rotamm15_grid(lon_d['amm15'], lat_d['amm15'])

            dlon_rotamm15 = (np.diff(lon_rotamm15)).mean()
            dlat_rotamm15 = (np.diff(lat_rotamm15)).mean()
            nlon_rotamm15 = lon_rotamm15.size
            nlat_rotamm15 = lat_rotamm15.size

            rot_dict[configd[2]] = {}
            rot_dict[configd[2]]['dlon'] = dlon_rotamm15
            rot_dict[configd[2]]['dlat'] = dlat_rotamm15
            rot_dict[configd[2]]['nlon'] = nlon_rotamm15
            rot_dict[configd[2]]['nlat'] = nlat_rotamm15
            rot_dict[configd[2]]['lon_rot'] = lon_rotamm15
            rot_dict[configd[2]]['lat_rot'] = lat_rotamm15 
            del(dlon_rotamm15)
            del(dlat_rotamm15)

            del(nlon_rotamm15)
            del(nlat_rotamm15)
            del(lon_rotamm15)
            del(lat_rotamm15)

    """
    #pdb.set_trace()

    '''
    NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out = ind_from_lon_lat('Dataset 2',configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1],lat_d[1])
    #NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out = ind_from_lon_lat('Dataset 1',configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[2],lat_d[2])
    regrid_params[tmp_datstr] = None,None, None, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out
    '''



    #default regirdding settings
    regrid_meth = 1 # NN
    regrid_params = None
    regrid_params = {}  
    regrid_params['do_regrid'] = 0


    # regridding indices.
    for tmp_datstr in Dataset_lst[1:]:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
        #rootgrp_gdept_dict[tmp_datstr] = rootgrp_gdept_dict['Dataset 1']
        

        regrid_params[tmp_datstr] = None#(None,None,None,None,None)

        if (configd[th_d_ind] is not None) & (configd[th_d_ind]!=configd[1]):
        #if (configd[th_d_ind] is not None) :   
            """

            if (configd[1].upper() in ['AMM7','AMM15']) & (configd[th_d_ind].upper() in ['AMM7','AMM15']):  
                #mesh_file_2nd = config_fnames_dict[configd[th_d_ind]]['mesh_file'] 
                #rootgrp_gdept_dict[tmp_datstr] = Dataset(mesh_file_2nd, 'r', format='NETCDF4')

                regrid_params['do_regrid'] = 2

                if (configd[1].upper() == 'AMM15') & (configd[th_d_ind].upper() == 'AMM7'):  


                    lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
                    lat = np.arange(40.066669,65+1/15.,1/15.)


                    amm_conv_dict = {}
                    rootgrp = Dataset(config_fnames_dict[configd[1]]['regrid_amm7_amm15'], 'r')
                    for var_conv in rootgrp.variables.keys(): amm_conv_dict[var_conv] = rootgrp.variables[var_conv][:]
                    rootgrp.close()
        
                    nlon_amm        = nlon_amm15
                    nlat_amm        = nlat_amm15
                    nlon_amm_2nd    = nlon_amm7
                    nlat_amm_2nd    = nlat_amm7



                elif (configd[1].upper() == 'AMM7') & (configd[th_d_ind].upper() == 'AMM15'):

                    amm_conv_dict = {}
                    rootgrp = Dataset(config_fnames_dict[configd[th_d_ind]]['regrid_amm15_amm7'], 'r')
                    for var_conv in rootgrp.variables.keys(): amm_conv_dict[var_conv] = rootgrp.variables[var_conv][:]
                    rootgrp.close()
        
                    nlon_amm        = nlon_amm7
                    nlat_amm        = nlat_amm7
                    nlon_amm_2nd    = nlon_amm15
                    nlat_amm_2nd    = nlat_amm15

                regrid_params[tmp_datstr] = regrid_2nd_thin_params(amm_conv_dict,nlon_amm,nlat_amm, nlon_amm_2nd,nlat_amm_2nd,thd)
                #pdb.set_trace()
                #pdb.set_trace()
            else:
                NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out = ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1],lat_d[1])




            """



            
            (NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, 
            NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out) = int_ind_wgt_from_xypos(tmp_datstr,
                                                    configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1],lat_d[1])
        
            regrid_params[tmp_datstr] = (NWS_amm_bl_jj_ind_out, NWS_amm_bl_ii_ind_out, NWS_amm_wgt_out, NWS_amm_nn_jj_ind_out, NWS_amm_nn_ii_ind_out)
            #pdb.set_trace()
            regrid_params['do_regrid'] = 2
    init_timer.append((datetime.now(),'AMM15 grid rotated'))

    # find variables common to both data sets, and use them for the buttons
    #pdb.set_trace()
    var_but_mat = var_d[1]['mat'].copy()
    # If two datasets, find variables in both datasets
    if load_second_files:   
        #pdb.set_trace()
        # (as Dataset 1 and 2 can be ORCA12 and 3 and 4 can be AMM15, need to expand:
        #   var_but_mat = np.intersect1d(var_d[1]['mat'], var_d[2]['mat'])
        var_but_mat = var_d[1]['mat'].copy()
        for th_d_ind in range(2,nDataset+1):var_but_mat = np.intersect1d(var_but_mat, var_d[th_d_ind]['mat'])
       
        
        
        # sort them to match the order of the first dataset
        var_but_mat_order = []
        for var_but in var_but_mat:var_but_mat_order.append(np.where(var_d[1]['mat'] == var_but )[0][0])
        var_but_mat = var_but_mat[np.argsort(var_but_mat_order)]

    nbutvar = var_but_mat.size


    #pdb.set_trace()

    
    init_timer.append((datetime.now(),'Found common vars all Datasets'))

    # set up figure.
    #   set up default figure, and then and and delete plots when you change indices.
    #   change indices with mouse click, detected with ginput
    #   ginput only works on one axes, so add a hidden fill screen axes, and then convert figure indices to an axes, and then using axes position and x/ylims into axes index. 
    #   create boxes with variable names as buttons to change variables. 
    climnorm = None # matplotlib.colors.LogNorm(0.005,0.1)
    

    print('Preparing Obs', datetime.now())
    #Set up Obs if in use
    if do_Obs:
        ob_ti = ti

        #Obs_hide = False
        Obs_hide_edges = False
        Obs_pair_loc = True
        Obs_AbsAnom = True
        

        #Available Obs data types
        Obs_varlst = Obs_dict['Dataset 1'].keys()

        # Obs data types to hide
        #Obs_obstype_hide = ['ProfT','ProfS','SST_ins','SST_sat','SLA', 'ChlA']
        #Obs_obstype_hide = ['SST_sat']

        # a dictionary of Obs datetimes, assuming midday of each day. 
        Obs_JULD_datetime_dict = {}
        for tmp_datstr in Dataset_lst:
            Obs_JULD_datetime_dict[tmp_datstr] = {} #{'ProfT':Obs_dict[tmp_datstr]['ProfT']['JULD']}
            for ob_var in Obs_varlst:
                Obs_JULD_datetime_dict[tmp_datstr][ob_var] = Obs_dict[tmp_datstr][ob_var]['JULD']


        # Obs plotting options 
        Obs_scatEC = None
        # size of markers
        Obs_scatSS = matplotlib.rcParams['lines.markersize'] ** 2
        Obs_scatSS = 100 # matplotlib.rcParams['lines.markersize'] ** 2
        Obs_scatSS = 250 # matplotlib.rcParams['lines.markersize'] ** 2

        Obs_vis_d = {}
        Obs_vis_d['Scat_symsize'] ={}
        Obs_vis_d['Scat_edgecol'] = {}
        Obs_vis_d['visible'] = {}
        for ob_var in Obs_varlst:   
            Obs_vis_d['Scat_symsize'][ob_var] = 250
            Obs_vis_d['Scat_edgecol'][ob_var] = 'k'

            Obs_vis_d['visible'][ob_var]= True
            if ob_var in ['SLA', 'ChlA', 'SST_sat']:
                Obs_vis_d['Scat_symsize'][ob_var] = 100
                Obs_vis_d['Scat_edgecol'][ob_var] = None
                Obs_vis_d['visible'][ob_var] = False


        Obs_vis_d['Prof_obs_col'] = 'k'
        Obs_vis_d['Prof_obs_ms'] = '.'
        Obs_vis_d['Prof_obs_ls'] = '-'
        Obs_vis_d['Prof_obs_lw'] = 2
        Obs_vis_d['Prof_obs_lw_2d'] = 1
        Obs_vis_d['Prof_obs_ls_2d'] = '--'

        Obs_vis_d['Prof_mod_col'] = 'm'
        Obs_vis_d['Prof_mod_ms'] = '.'
        Obs_vis_d['Prof_mod_ls'] = '-'
        Obs_vis_d['Prof_mod_lw'] = 2
        Obs_vis_d['Prof_mod_lw_2d'] = 1
        Obs_vis_d['Prof_mod_ls_2d'] = '--'

        # set up empty profile dictionaries for plotting
        (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
            obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst)
        '''
        obs_z_sel = {}
        obs_obs_sel = {}
        obs_mod_sel = {}
        obs_lon_sel = {}
        obs_lat_sel = {}

        obs_stat_id_sel = {}
        obs_stat_type_sel = {}
        obs_stat_time_sel = {}
                    

        for tmp_datstr in Dataset_lst:
            obs_z_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
            obs_obs_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
            obs_mod_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
            obs_lon_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
            obs_lat_sel[tmp_datstr] = np.ma.zeros((1))*np.ma.masked
            obs_stat_id_sel[tmp_datstr] = ''
            obs_stat_time_sel[tmp_datstr] = ''
            obs_stat_type_sel[tmp_datstr] = None
        '''

        # set reload Obs to true
        reload_Obs = True

        init_timer.append((datetime.now(),'Obs Loaded '))

    print('Creating Figure', datetime.now())
    ax = []

    fig_tit_str = 'Interactive figure, Select lat/lon in a); lon in b); lat  in c); depth in d) and time in e).\n'
    #if fig_lab_d['Dataset 1'] is not None: fig_tit_str = fig_tit_str + ' Dataset 1 = %s;'%fig_lab_d['Dataset 1']
    #if fig_lab_d['Dataset 2'] is not None: fig_tit_str = fig_tit_str + ' Dataset 2 = %s;'%fig_lab_d['Dataset 2']

    for tmp_datstr in Dataset_lst:
        #if fig_lab_d[tmp_datstr] is not None: 
        fig_tit_str = fig_tit_str + ' %s = %s;'%(tmp_datstr,fig_lab_d[tmp_datstr])


    fig_tit_str_int = 'Interactive figure, Select lat/lon in a); lon in b); lat  in c); depth in d) and time in e). %s[%i, %i, %i, %i] (thin = %i; thin_files = %i) '%(var,ii,jj,zz,ti, thd[1]['dx'], thd[1]['df'])
    fig_tit_str_lab = ''
    #if fig_lab_d['Dataset 1'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 1 = %s;'%fig_lab_d['Dataset 1']
    #if fig_lab_d['Dataset 2'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 2 = %s;'%fig_lab_d['Dataset 2']
    for tmp_datstr in Dataset_lst:
        #if fig_lab_d[tmp_datstr] is not None: 
        fig_tit_str_lab = fig_tit_str_lab + ' %s = %s;'%(tmp_datstr,fig_lab_d[tmp_datstr])


    nvarbutcol = 16 # 18
    nvarbutcol = 22 # 18
    nvarbutcol = 25 # 18


    if justplot:
        nvarbutcol = 1000



    init_timer.append((datetime.now(),'Fig str, n buttons'))
    #import locale
    #locale.setlocale(locale.LC_ALL, 'en_GB.utf8')


    fig = plt.figure()
    fig.suptitle(fig_tit_str_int + '\n' + fig_tit_str_lab, fontsize=figsuptitfontsize)
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    if nbutvar <nvarbutcol:
        plt.subplots_adjust(top=0.88,bottom=0.1,left=0.09,right=0.91,hspace=0.2,wspace=0.065)
    #elif nbutvar >(nvarbutcol):
    elif (nbutvar >=(nvarbutcol))&(nbutvar <(2*nvarbutcol)):
        plt.subplots_adjust(top=0.88,bottom=0.1,left=0.15,right=0.91,hspace=0.2,wspace=0.065)
    elif nbutvar >=2*(nvarbutcol):
        plt.subplots_adjust(top=0.88,bottom=0.1,left=0.21,right=0.91,hspace=0.2,wspace=0.065)

    #width = leftgap + axwid + wgap + axwid = right (0.91)

    cbwid,cbgap = 0.01,0.01
    wgap = 0.06
    hgap = 0.04
    dyhig = 0.17
    axwid = 0.4
    if nbutvar <nvarbutcol: # 9 + 39 + 6 + 39
        axwid = 0.38 #0.39
        leftgap = 0.09
    #else:
    elif (nbutvar >=(nvarbutcol))&(nbutvar <(2*nvarbutcol)): # 15 + 35 + 6 + 35
        axwid = 0.35
        leftgap = 0.15
    elif nbutvar >=(2*nvarbutcol): # 15 + 35 + 6 + 35
        axwid = 0.32
        leftgap = 0.21

    profwid = 0.1
    profgap = 0.04
    profvis = True
    ax_position_dims = []
    ax_position_dims.append([leftgap,                                  0.10, axwid - cbwid - cbgap,  0.80])
    ax_position_dims.append([leftgap + (axwid - cbwid - cbgap) + wgap, 0.73, axwid - cbwid - cbgap,  0.17])
    ax_position_dims.append([leftgap + (axwid - cbwid - cbgap) + wgap, 0.52, axwid - cbwid - cbgap,  0.17])
    ax_position_dims.append([leftgap + (axwid - cbwid - cbgap) + wgap, 0.31, axwid - cbwid - cbgap,  0.17])
    ax_position_dims.append([leftgap + (axwid - cbwid - cbgap) + wgap, 0.10, axwid - cbwid - cbgap,  0.17])
    #
    ax_position_dims.append([0,0,0,0])
    #
    
    ax_position_dims_prof = []
    ax_position_dims_prof.append([leftgap,                                             0.10, axwid - cbwid - cbgap,  0.80])
    ax_position_dims_prof.append([leftgap + (axwid - cbwid - cbgap) + wgap+ profwid + profgap, 0.73, axwid - cbwid - cbgap - profwid - profgap,  0.17])
    ax_position_dims_prof.append([leftgap + (axwid - cbwid - cbgap) + wgap+ profwid + profgap, 0.52, axwid - cbwid - cbgap - profwid - profgap,  0.17])
    ax_position_dims_prof.append([leftgap + (axwid - cbwid - cbgap) + wgap,            0.31, axwid - cbwid - cbgap,  0.17])
    ax_position_dims_prof.append([leftgap + (axwid - cbwid - cbgap) + wgap,            0.10, axwid - cbwid - cbgap,  0.17])
    #
    ax_position_dims_prof.append([leftgap + (axwid - cbwid - cbgap) + wgap, 0.52, profwid,  0.73 + 0.17 - 0.52])
    

    if profvis:
        for tmpax_position_dims in ax_position_dims_prof:  ax.append(fig.add_axes(tmpax_position_dims))
    else:
        for tmpax_position_dims in ax_position_dims:  ax.append(fig.add_axes(tmpax_position_dims))
    ax[-1].set_visible(profvis)

    #ax.append(fig.add_axes([leftgap,                                  0.10, axwid - cbwid - cbgap,  0.80]))
    #ax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap, 0.73, axwid - cbwid - cbgap,  0.17]))
    #ax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap, 0.52, axwid - cbwid - cbgap,  0.17]))
    #ax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap, 0.31, axwid - cbwid - cbgap,  0.17]))
    #ax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap, 0.10, axwid - cbwid - cbgap,  0.17]))


    print('Creating Figure', datetime.now())


    labi,labj = 0.05, 0.95
    for ai,tmpax in enumerate(ax): tmpax.text(labi,labj,'%s)'%letter_mat[ai], transform=tmpax.transAxes, ha = 'left', va = 'top', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none'))
           
    init_timer.append((datetime.now(),'Created Figure and axes, fig letters'))

    tsaxtx_lst = []
    tsaxtxd_lst = []

    if nDataset == 1:
        tsaxtx_lst.append(ax[4].text(0.01,0.01,fig_lab_d['Dataset 1'], ha = 'left', va = 'bottom', transform=ax[4].transAxes, color = 'r', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))

    elif nDataset ==2:
        tsaxtx_lst.append(ax[4].text(0.01,0.01,fig_lab_d['Dataset 1'], ha = 'left', va = 'bottom', transform=ax[4].transAxes, color = 'r', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
        tsaxtx_lst.append(ax[4].text(0.99,0.01,fig_lab_d['Dataset 2'], ha = 'right', va = 'bottom', transform=ax[4].transAxes, color = 'b', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))

        tsaxtxd_lst.append(ax[4].text(0.99,0.975,'Dat2-Dat1', ha = 'right', va = 'top', transform=ax[4].transAxes, color = 'g', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
    
    else:
        tmp_vlist = np.linspace(0.85,0.15,nDataset)
        for tdsi,tmp_datstr in enumerate(Dataset_lst):
            tsaxtx_lst.append(ax[4].text(0.01,tmp_vlist[tdsi],fig_lab_d[tmp_datstr], ha = 'left', va = 'top', transform=ax[4].transAxes, color = Dataset_col[tdsi], fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
        del(tmp_vlist)

        
        tmp_vlist_anom = np.linspace(0.85,0.15,nDataset*(nDataset-1))
        tmp_vlist_anom_cnt = 0
        for tdsi1,tmp_datstr1 in enumerate(Dataset_lst):
            for tdsi2,tmp_datstr2 in enumerate(Dataset_lst):
                if tdsi1 != tdsi2:
                    
                    tmp_anom_str = 'Dat%i-Dat%i'%(tdsi1+1,tdsi2+1)
                    
                    tsaxtxd_lst.append(ax[4].text(0.99,tmp_vlist_anom[tmp_vlist_anom_cnt],tmp_anom_str, ha = 'right', va = 'top', transform=ax[4].transAxes, color = Dataset_col_diff_dict[tmp_anom_str], fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
                    tmp_vlist_anom_cnt += 1
    


    for tmp_datstr in Dataset_lst:        
        fig_tit_str_lab = fig_tit_str_lab + ' %s = %s;'%(tmp_datstr,fig_lab_d[tmp_datstr])


    init_timer.append((datetime.now(),'Figure, adding text'))

    #flip depth axes
    for tmpax in ax[1:]: tmpax.invert_yaxis()
    #use log depth scale, setiched off as often causes problems (clashes with hidden axes etc).
    #for tmpax in ax[1:]: tmpax.set_yscale('log')

    # add hidden fill screen axes 
    clickax = fig.add_axes([0,0,1,1], frameon=False)
    clickax.axis('off')
    

    
    init_timer.append((datetime.now(),'Figure Created'))


    if verbose_debugging: print('Created figure', datetime.now())

    #pdb.set_trace()
    #add "buttons"
    but_x0 = 0.01
    but_x1 = 0.06
    func_but_x1 = 0.99
    func_but_x0 = 0.94
    func_but_dx1 = func_but_x1 -func_but_x0 
    but_dy = 0.04
    but_dy = 0.03
    but_dy = 0.025
    but_ysp = 0.01 
    but_ysp = 0.01 
    
    but_dysp = but_dy + but_ysp 
    

    but_extent = {}
    but_line_han,but_text_han = {},{}
    for vi,var_dat in enumerate(var_but_mat): 
        tmpcol = 'k'
        if var_dim[var_dat] == 3: #  2D (+time) vars 
            if var_grid['Dataset 1'][var_dat] == 'T': tmpcol = 'deepskyblue'
            if var_grid['Dataset 1'][var_dat] == 'U': tmpcol = 'yellow'
            if var_grid['Dataset 1'][var_dat] == 'V': tmpcol = 'yellow'
            if var_grid['Dataset 1'][var_dat] == 'WW3': tmpcol = 'lightsteelblue'
            if var_grid['Dataset 1'][var_dat] == 'I': tmpcol = 'lightgreen'
        if var_dim[var_dat] == 4: #  3D (+time) vars 
            if var_grid['Dataset 1'][var_dat] == 'T': tmpcol = 'b'
            if var_grid['Dataset 1'][var_dat] == 'U': tmpcol = 'gold'
            if var_grid['Dataset 1'][var_dat] == 'V': tmpcol = 'gold'
            if var_grid['Dataset 1'][var_dat] == 'WW3': tmpcol = 'navy'
            if var_grid['Dataset 1'][var_dat] == 'I': tmpcol = 'darkgreen'
        if var_dat in var_d['d']: tmpcol = '0.5'
        vi_num = vi
        if (vi>=nvarbutcol)&( vi<2*nvarbutcol):
            vi_num = vi-nvarbutcol

            but_x0 = 0.01 + 0.06
            but_x1 = 0.06 + 0.06
        elif (vi>=2*nvarbutcol):
            vi_num = vi-nvarbutcol-nvarbutcol

            but_x0 = 0.01 + 0.06 + 0.06
            but_x1 = 0.06 + 0.06 + 0.06
      

        #note button extends (as in position.x0,x1, y0, y1)
        #but_extent[var_dat] = np.array([but_x0,but_x1,0.9 - (but_dy + vi*but_dysp),0.9 - (0 + vi_num*but_dysp)])
        but_extent[var_dat] = np.array([but_x0,but_x1,0.9 - (but_dy + (vi_num*but_dysp)), 0.9 - (vi_num*but_dysp)])
        #add button box
        but_line_han[var_dat] = clickax.plot([but_x0,but_x1,but_x1,but_x0,but_x0],0.9 - (np.array([0,0,but_dy,but_dy,0]) + vi_num*but_dysp),color = tmpcol)
        #add button names
        but_text_han[var_dat] = clickax.text((but_x0+but_x1)/2,0.9 - ((but_dy/2) + vi_num*but_dysp),var_dat, ha = 'center', va = 'center')



    clickax.axis([0,1,0,1])
    

    if verbose_debugging: print('Added variable boxes', datetime.now())

    mode_name_lst = ['Click','Loop']

    func_names_lst = ['Hov/Time','Show Prof',
                      'Zoom',
                      'Axis', 'ColScl', 'Clim: Zoom','Clim: pair','Clim: sym',
                      'Surface', 'Near-Bed', 'Surface-Bed','Depth-Mean','Depth level',
                      'Contours','Grad','Time Diff','Sec Grid','TS Diag','LD time','Fcst Diag','Vis curr','MLD','Obs','Xsect','Save Figure','Help','Quit'] 
    #'Reset zoom','Obs: sel','Obs: opt','Clim: Reset','Clim: Expand',
    
    Sec_regrid = False
    if uniqconfig.size==1:
        func_names_lst.remove('Sec Grid')


    do_Xsect = True
    loaded_xsect = False
    if not do_Xsect:
        
        func_names_lst.remove('Xsect')
    else:
        figxs = None



    
    if not do_MLD:        
        func_names_lst.remove('MLD')
    else:
        reload_MLD = True
        MLD_show = True
        poss_MLD_var_lst_lower = ['mld','karamld','mld25h_1','mld25h_2','somxzint','somxl010','sokaraml']
        MLD_var_lst = [ss for ss in var_but_mat if ss.lower() in poss_MLD_var_lst_lower]
        if len(MLD_var_lst)>0:
            MLD_var = MLD_var_lst[0] # 'mld25h_1'
            data_mld = {}
            mldax_lst = []
        else:
            do_MLD = False      
            func_names_lst.remove('MLD')
            reload_MLD = False
            MLD_show = False
        #figmlopt = None


    # if Obs, create empty option fiugre handle, otherwise remove button names
    if not do_Obs:
        #func_names_lst.remove('Obs: sel')
        #func_names_lst.remove('Obs: opt')
        func_names_lst.remove('Obs')
    #else:
        #figobsopt = None
        

    if add_TSProf:
        #ts_diag_coord = np.ma.ones(3)*np.ma.masked
        figts = None
        figfc = None
    else:
        func_names_lst.remove('TS Diag')

    if nldi < 2: # no point being able to change lead time database if only one 
        func_names_lst.remove('LD time')
        func_names_lst.remove('Fcst Diag')
    else:
        ldi=0

    if vis_curr == -1:
        func_names_lst.remove('Vis curr')

    # For T Diff
    if ntime < 2: # no point being able to change lead time database if only one 
        func_names_lst.remove('Time Diff')
        do_Tdiff = False
    else:
        do_Tdiff = True

        Time_Diff = False
        data_inst_Tm1 = {}
        #data_inst_Tm1['Dataset 1'],data_inst_Tm1['Dataset 2'] = None,None
        preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1 = 0.5,'None',0.5

        

    if load_second_files == False:
        func_names_lst.remove('Clim: pair')

    func_names_lst = func_names_lst + mode_name_lst

    # if a secondary data set, give ability to change data sets. 
    if load_second_files:
        func_names_lst = func_names_lst + secdataset_proc_list
        if regrid_params['do_regrid'] == 2:
            func_names_lst = func_names_lst + ['regrid_meth']
        
        if do_ensemble: func_names_lst = func_names_lst + ens_stat_lst 

    func_but_line_han,func_but_text_han = {},{}
    func_but_extent = {}
    

    mode_name_secdataset_proc_list = mode_name_lst

    if load_second_files: 
        mode_name_secdataset_proc_list = mode_name_secdataset_proc_list + secdataset_proc_list 
        if do_ensemble: mode_name_secdataset_proc_list = mode_name_secdataset_proc_list + ens_stat_lst        
        if regrid_params['do_regrid'] == 2:
            mode_name_secdataset_proc_list = mode_name_secdataset_proc_list + ['regrid_meth']




    #add default button box
    #   overwritten for mode_name_secdataset_proc_list
    for vi,funcname in enumerate(func_names_lst): 

        #note button extends (as in position.x0,x1, y0, y1)
        #func_but_extent[funcname] = [func_but_x0,func_but_x1,0.95 - (but_dy + vi*0.05),0.95 - (0 + vi*0.05)]
        func_but_extent[funcname] = [func_but_x0,func_but_x1,0.90 - (but_dy + vi*but_dysp),0.90 - (0 + vi*but_dysp)]

    '''
    for vi, tmp_funcname in enumerate(mode_name_secdataset_proc_list):
        func_but_extent[tmp_funcname] = [0.15 + vi*(func_but_dx1+0.01), 0.15 + vi*(func_but_dx1+0.01) + func_but_dx1, 0.025,  0.025 + but_dy]
    '''

    # to allow Dataset and Dat-Dat buttons size to vary depending on the number of data sets
    # button width and gap vary, with a default value of tmpfunc_but_dx1,tmp_dx_gap = 0.05, 0.01 
    # iterate through buttons
    # start with a left position of if vi == 0:tmp_lhs = 0.15, 
    #       let tmp_rhs = tmp_lhs + tmpfunc_but_dx1, and then tmp_lhs = tmp_rhs + gap.
    # gap and button size depend on the number of datasets (nDataset), and whether dataset of dat-dat

    tmp_bot = 0.025
    tmp_top = 0.05
    if nDataset > 8:
        tmp_bot = 0.035
        tmp_top = 0.06

    for vi, tmp_funcname in enumerate(mode_name_secdataset_proc_list):

        tmpfunc_but_dx1 = 0.05# func_but_dx1
        tmp_dx_gap = 0.01
        if (tmp_funcname in secdataset_proc_list):
            if (tmp_funcname not in Dataset_lst):

                if do_ensemble: continue
                
                if (nDataset>6):
                    tmp_datmdat_str_lst = tmp_funcname.split('-')     #['Dat1', 'Dat3']
                    if int(tmp_datmdat_str_lst[0][3:])>int(tmp_datmdat_str_lst[1][3:]): continue
                    
            
                if nDataset in [3,4]:
                    tmpfunc_but_dx1 = 0.025# 0.05/2
                elif nDataset in [5,6]:
                    tmpfunc_but_dx1 = 0.0125# 0.05/4
                    '''
                if nDataset>6:
                    tmpfunc_but_dx1 = 0.015
                    tmp_dx_gap = 0.0025
            else:
                if nDataset>6:
                    tmpfunc_but_dx1 = 0.015
                    tmp_dx_gap = 0.0025
                    '''

            if nDataset>4:
                tmpfunc_but_dx1 = np.minimum(tmpfunc_but_dx1,0.04)
                tmp_dx_gap = 0.0025
            if nDataset>6:
                tmpfunc_but_dx1 = 0.015
                tmp_dx_gap = 0.0025


        if tmp_funcname == 'Dataset 1': tmp_dx_gap = 0.01

        if vi == 0: 
            tmp_lhs = 0.15
        else:
            tmp_lhs = tmp_rhs + tmp_dx_gap
        tmp_rhs = tmp_lhs + tmpfunc_but_dx1

        if tmp_rhs>0.99:

            tmp_lhs = 0.15
            tmp_rhs = tmp_lhs + tmpfunc_but_dx1

            tmp_bot = 0.002
            tmp_top = 0.027
            tmp_bot = 0.005
            tmp_top = 0.03


        func_but_extent[tmp_funcname] = [tmp_lhs, tmp_rhs, tmp_bot,  tmp_top]



    #pdb.set_trace()

    del(tmp_lhs)
    del(tmp_rhs)
    del(tmp_top)
    del(tmp_bot)

    for vi,funcname in enumerate(func_names_lst): 
        if funcname not in func_but_extent.keys(): continue
        #add button outlines
        func_but_line_han[funcname] = clickax.plot([func_but_extent[funcname][0],func_but_extent[funcname][1],func_but_extent[funcname][1],func_but_extent[funcname][0],func_but_extent[funcname][0]], [func_but_extent[funcname][2],func_but_extent[funcname][2],func_but_extent[funcname][3],func_but_extent[funcname][3],func_but_extent[funcname][2]],'k')
        #add button names
        func_but_text_han[funcname] = clickax.text((func_but_extent[funcname][0]+func_but_extent[funcname][1])/2,(func_but_extent[funcname][2]+func_but_extent[funcname][3])/2,funcname, ha = 'center', va = 'center')
    
    if nDataset>2:
        for tmp_funcname in secdataset_proc_list: # [...'Dataset 3', 'Dataset 4', 'Dat1-Dat2', 'Dat1-Dat3', 'Dat1-Dat4', ...]

            tmp_datmdat_str = func_but_text_han[tmp_funcname].get_text() #'Dat1-Dat2'
            new_tmp_datmdat_str = tmp_datmdat_str
            if (tmp_funcname not in Dataset_lst): #[...'Dataset 2', 'Dataset 3', 'Dataset 4', ...]
                
                tmp_datmdat_str_lst = tmp_datmdat_str.split('-')     #['Dat1', 'Dat3']

                if nDataset in [3,4]:
                    new_tmp_datmdat_str = 'D%i-D%i'%(int(tmp_datmdat_str_lst[0][3:]),int(tmp_datmdat_str_lst[1][3:])) # 'D%i-D%i'%(1,3)
                elif nDataset >4:
                    new_tmp_datmdat_str = '%i-%i'%(int(tmp_datmdat_str_lst[0][3:]),int(tmp_datmdat_str_lst[1][3:])) # 'D%i-D%i'%(1,3)
                
            else:
                if nDataset>6:
                    new_tmp_datmdat_str = 'D%i'%int(tmp_datmdat_str[8:])

            func_but_text_han[tmp_funcname].set_text(new_tmp_datmdat_str)

             

    # if a secondary data set, det default behaviour. 
    if load_second_files: func_but_text_han[secdataset_proc].set_color('darkgreen')

    func_but_text_han['waiting'] = clickax.text(0.01,0.995,'Working', color = 'r', va = 'top') # clwaithandles
    
    # Set intial mode to be Click
    func_but_text_han['Click'].set_color('gold')

    func_but_text_han['Depth level'].set_color('k')
    func_but_text_han['Surface'].set_color('k')
    func_but_text_han['Near-Bed'].set_color('k')
    func_but_text_han['Surface-Bed'].set_color('k')
    func_but_text_han['Depth-Mean'].set_color('k')
    if z_meth == 'z_slice':func_but_text_han['Depth level'].set_color('r')
    if z_meth == 'ss':func_but_text_han['Surface'].set_color('r')
    if z_meth == 'nb':func_but_text_han['Near-Bed'].set_color('r')
    if z_meth == 'df':func_but_text_han['Surface-Bed'].set_color('r')
    if z_meth == 'zm':func_but_text_han['Depth-Mean'].set_color('r')

    func_but_text_han['waiting'].set_color('w')

    
    if load_second_files: 
        if regrid_params['do_regrid'] == 2:
            func_but_text_han['regrid_meth'].set_text('Regrid: NN')
        if clim_pair:func_but_text_han['Clim: pair'].set_color('gold')

    if hov_time:
        func_but_text_han['Hov/Time'].set_color('darkgreen')
    else:
        func_but_text_han['Hov/Time'].set_color('0.5')
    if profvis:
        func_but_text_han['Show Prof'].set_color('darkgreen')
    else:
        func_but_text_han['Show Prof'].set_color('0.5')



    if do_cont:
        func_but_text_han['Contours'].set_color('darkgreen')
    else:
        func_but_text_han['Contours'].set_color('0.5')

    if do_grad == 1:
        func_but_text_han['Grad'].set_color('darkgreen')
    else:
        func_but_text_han['Grad'].set_color('0.5')
        func_but_text_han['Grad'].set_text('Grad')


    reload_UV_map = False
    if vis_curr == 0:
        func_but_text_han['Vis curr'].set_color('k')
        reload_UV_map = False
    elif vis_curr > 0:
        reload_UV_map = True
        func_but_text_han['Vis curr'].set_color('darkgreen')

    func_but_text_han['ColScl'].set_text('Col: Linear')

    func_but_text_han['Axis'].set_text('Axis: Auto')


    # Buttons that can be right clicked for different behavour have a douple outline. 
    str_pe = [pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()]
    func_but_line_han['Zoom'][0].set_linewidth(1)
    func_but_line_han['Zoom'][0].set_color('w')
    func_but_line_han['Zoom'][0].set_path_effects(str_pe)
    func_but_line_han['Clim: Zoom'][0].set_linewidth(1)
    func_but_line_han['Clim: Zoom'][0].set_color('w')
    func_but_line_han['Clim: Zoom'][0].set_path_effects(str_pe)
    if do_Obs:
        #pdb.set_trace()
        func_but_line_han['Obs'][0].set_linewidth(1)
        func_but_line_han['Obs'][0].set_color('w')
        func_but_line_han['Obs'][0].set_path_effects(str_pe)
    if do_Xsect:
        #pdb.set_trace()
        func_but_line_han['Xsect'][0].set_linewidth(1)
        func_but_line_han['Xsect'][0].set_color('w')
        func_but_line_han['Xsect'][0].set_path_effects(str_pe)
    if do_MLD:
        #pdb.set_trace()
        func_but_line_han['MLD'][0].set_linewidth(1)
        func_but_line_han['MLD'][0].set_color('w')
        func_but_line_han['MLD'][0].set_path_effects(str_pe)
    #pdb.set_trace()


    ldi = 0
    if nldi > 2:
        func_but_text_han['LD time'].set_text('LD time: %s'%ld_lab_mat[ldi])


    init_timer.append((datetime.now(),'Added functions boxes'))

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

    init_timer.append((datetime.now(),'Added Mouse tracking functions'))



    global mouse_info

    #pdb.set_trace()
    def onclick(event):
        '''
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        ('double' if event.dblclick else 'single', event.button,
        event.x, event.y, event.xdata, event.ydata))
        '''

        global mouse_info
        mouse_info = {'button':event.button,'x':event.x, 'y':event.y, 'xdata':event.xdata, 'ydata':event.ydata}
        #fig.canvas.mpl_disconnect(cid)

        return event.button, event.x, event.y, event.xdata, event.ydata


    mouse_click_id = fig.canvas.mpl_connect('button_press_event', onclick)


    but_text_han[var].set_color('r')

    if verbose_debugging: print('Added functions boxes', datetime.now())


    ###########################################################################
    # Define inner functions
    ###########################################################################

    #global map_x,map_y,map_dat,ew_slice_dict['x'],ew_slice_dict['y'],ew_slice_dat,ns_slice_dict['x'],ns_slice_dict['y'],ns_slice_dat,hov_x,hov_y,hov_dat,ts_x,ts_dat
    #global ii,jj

    #

    if verbose_debugging: print('Create inner functions', datetime.now())
    init_timer.append((datetime.now(),'Create inner functions'))


    def indices_from_ginput_ax(ax,clii,cljj,thd,ew_line_x = None,ew_line_y = None,ns_line_x = None,ns_line_y = None):


        '''
        ginput doesn't tell you which subplot you are clicking, only the position within that subplot.
        we need which axis is clicked as well as the cooridinates within that axis
        
        we therefore trick ginput to give use figure coordinate (with a dummy, invisible full figure size subplot
        in front of everything, and then use this function to turn those coordinates into the coordinates within the 
        the subplot, and the which axis/subplot it is ax,

        NB   indices_from_ginput_ax hard coded to return indices for Dataset 1.
            because uses configd[1], and lon_d[1]

        '''
        sel_ii,sel_jj,sel_ti ,sel_zz = None,None,None,None
        sel_ax = None
        xlocval, ylocval  = None, None
        for ai,tmpax in enumerate(ax): # ai = 0; tmpax = ax[ai]
            tmppos =  tmpax.get_position()
            # was click within extent
            if (clii >= tmppos.x0) & (clii <= tmppos.x1) & (cljj >= tmppos.y0) & (cljj <= tmppos.y1):
                sel_ax = ai

                #convert figure coordinate of click, into location with the axes, using data coordinates
                clxlim = np.array(tmpax.get_xlim())
                clylim = np.array(tmpax.get_ylim())
                normxloc = (clii - tmppos.x0 ) / (tmppos.x1 - tmppos.x0)
                normyloc = (cljj - tmppos.y0 ) / (tmppos.y1 - tmppos.y0)
                #xlocval = normxloc*clxlim.ptp() + clxlim.min()
                #ylocval = normyloc*clylim.ptp() + clylim.min()
                xlocval = normxloc*np.ptp(clxlim) + clxlim.min()
                ylocval = normyloc*np.ptp(clylim) + clylim.min()
                """
                if (thd[1]['dx'] != 1):
                    if configd[1].upper() not in ['AMM7','AMM15', 'CO9P2', 'ORCA025','ORCA025EXT','GULF18','ORCA12','ORCA025ICE','ORCA12ICE']:
                        print('Thinning lon lat selection not programmed for ', configd[1].upper())
                        pdb.set_trace()
                """

                # what do the local coordiantes of the click mean in terms of the data to plot.
                # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.

                # if in map, covert lon lat to ii,jj
                if ai == 0:
                    loni,latj= xlocval,ylocval
                    '''

                    if configd[1].upper() in ['AMM7','GULF18']:
                        sel_ii = (np.abs(lon[thd[1]['x0']:thd[1]['x1']:thd[1]['dx']] - loni)).argmin()
                        sel_jj = (np.abs(lat[thd[1]['y0']:thd[1]['y1']:thd[1]['dy']] - latj)).argmin()
                    elif configd[1].upper() in ['AMM15','CO9P2']:
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        sel_ii = np.minimum(np.maximum( np.round((lon_mat_rot - lon_rotamm15[thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].min())/(dlon_rotamm15*thd[1]['dx'])).astype('int') ,0),nlon_rotamm15//thd[1]['dx']-1)
                        sel_jj = np.minimum(np.maximum( np.round((lat_mat_rot - lat_rotamm15[thd[1]['y0']:thd[1]['y1']:thd[1]['dy']].min())/(dlat_rotamm15*thd[1]['dx'])).astype('int') ,0),nlat_rotamm15//thd[1]['dx']-1)
                    elif configd[1].upper() in ['ORCA025','ORCA025EXT','ORCA12','ORCA025ICE','ORCA12ICE']:
                        sel_dist_mat = np.sqrt((lon_d[1][:,:] - loni)**2 + (lat_d[1][:,:] - latj)**2 )
                        sel_jj,sel_ii = sel_dist_mat.argmin()//sel_dist_mat.shape[1], sel_dist_mat.argmin()%sel_dist_mat.shape[1]

                    else:
                        print('config not supported:', configd[1])
                        pdb.set_trace()

                    #pdb.set_trace()
                    '''

                    print(sel_ii,sel_jj)
                    
                    sel_jj,sel_ii = ind_from_lon_lat('Dataset 1',configd,xypos_dict, lon_d,lat_d, thd,rot_dict,loni,latj)

                    if (sel_ii<0)|(sel_jj<0):
                        pdb.set_trace()
                    #domsize[th_d_ind]. domsize[1]
                    if (sel_ii>=domsize[1][1]):
                        #print('ii too big')
                        #pdb.set_trace()
                        sel_ii=domsize[1][1]-1
                    if (sel_jj>=domsize[1][0]):
                        #print('jj too big')
                        #pdb.set_trace()
                        sel_jj=domsize[1][0]-1
                    if (sel_jj<0):sel_jj=0
                    if (sel_ii<0):sel_ii=0
                    if (sel_jj<0):sel_jj=0
                        
                    '''
                    if (sel_ii>=lon_d[1].shape[1]):
                        #print('ii too big')
                        #pdb.set_trace()
                        sel_ii=lon_d[1].shape[1]-1
                    if (sel_jj>=lon_d[1].shape[0]):
                        #print('jj too big')
                        #pdb.set_trace()
                        sel_jj=lon_d[1].shape[0]-1
                    if (sel_jj<0):sel_jj=0
                    if (sel_ii<0):sel_ii=0
                    if (sel_jj<0):sel_jj=0
                    '''
                    # and reload slices, and hovmuller/time series

                elif ai in [1]: 
                    # if in ew slice, change ns slice, and hov/time series
                    loni= xlocval
                    sel_ii = (np.abs(ew_line_x - loni)).argmin()

                    '''
                    if configd[1].upper() == 'AMM7':
                        sel_ii = (np.abs(lon[thd[1]['x0']:thd[1]['x1']:thd[1]['dx']] - loni)).argmin()
                    elif configd[1].upper() in ['AMM15','CO9P2']:
                        latj =  ew_line_y[(np.abs(ew_line_x - loni)).argmin()] 
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        #sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - lon_rotamm15[thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].min())/(dlon_rotamm15*thd[1]['dx'])).astype('int'),0),nlon_rotamm15//thd[1]['dx']-1)
                        sel_ii = np.minimum(np.maximum(np.round((lon_mat_rot - rot_dict[configd[1]]['lon_rot'][thd[1]['x0']:thd[1]['x1']:thd[1]['dx']].min())/(rot_dict[configd[1]]['dlon']*thd[1]['dx'])).astype('int'),0),rot_dict[configd[1]]['nlon']//thd[1]['dx']-1)

                    elif configd[1].upper() in ['ORCA025','ORCA025EXT','ORCA12','ORCA025ICE','ORCA12ICE']:
                        sel_ii = (np.abs(ew_line_x - loni)).argmin()
                    else:
                        print('config not supported:', configd[1])
                        pdb.set_trace()
                    '''
                    #sel_zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )
                    sel_zz = int( (1-normyloc)*np.ptp(clylim) + clylim.min() )
                    
                    
                elif ai in [2]:
                    # if in ns slice, change ew slice, and hov/time series
                    latj= xlocval
                    sel_jj = (np.abs(ns_line_y - latj)).argmin()
                    '''
                    if configd[1].upper() == 'AMM7':
                        sel_jj = (np.abs(lat[thd[1]['y0']:thd[1]['y1']:thd[1]['dy']] - latj)).argmin()
                    elif configd[1].upper() in ['AMM15','CO9P2']:                        
                        loni =  ns_line_x[(np.abs(ns_line_y - latj)).argmin()]
                        lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(loni,latj)
                        #sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - lat_rotamm15[thd[1]['y0']:thd[1]['y1']:thd[1]['dy']].min())/(dlat_rotamm15*thd[1]['dx'])).astype('int'),0),nlat_rotamm15//thd[1]['dx']-1)
                        sel_jj = np.minimum(np.maximum(np.round((lat_mat_rot - rot_dict[configd[1]]['lat_rot'][thd[1]['y0']:thd[1]['y1']:thd[1]['dy']].min())/(rot_dict[configd[1]]['dlon']*thd[1]['dx'])).astype('int'),0),rot_dict[configd[1]]['nlat']//thd[1]['dx']-1)
                    elif configd[1].upper() in ['ORCA025','ORCA025EXT','ORCA12','ORCA025ICE','ORCA12ICE']:
                        sel_jj = (np.abs(ns_line_y - latj)).argmin()
                    else:
                        print('config not supported:', configd[1])
                        #pdb.set_trace()
                        '''
                    #sel_zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )
                    sel_zz = int( (1-normyloc)*np.ptp(clylim) + clylim.min() )

                elif ai in [3]:
                    # if in hov/time series, change map, and slices

                    # re calculate depth values, as y scale reversed, 
                    #sel_zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )
                    sel_zz = int( (1-normyloc)*np.ptp(clylim) + clylim.min() )
                    #pdb.set_trace()


                elif ai in [4]:
                    # if in hov/time series, change map, and slices
                    sel_ti = np.abs(xlocval - time_datetime_since_1970).argmin()
                    
                elif ai in [5]:
                    print('No action for Profiles axes')
                else:
                    print('clicked in another axes??')
                    return
                    pdb.set_trace()


        
        #return sel_ax,sel_ii,sel_jj,sel_ti,sel_zz
        return sel_ax,sel_ii,sel_jj,sel_ti,sel_zz,xlocval,ylocval



    def save_figure_funct():




        figdpi = 90
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        secdataset_proc_figname = ''

        if secdataset_proc in Dataset_lst:
            secdataset_proc_figname = '%s'%(secdataset_proc.replace('Dataset ','Datset_'))
        else:
            tmpdatasetnum_1 = secdataset_proc[3]
            tmpdatasetnum_2 = secdataset_proc[8]
            tmpdataset_oper = secdataset_proc[4]
            if tmpdataset_oper == '-':
                secdataset_proc_figname = 'Diff_%s-%s'%(tmpdatasetnum_1,tmpdatasetnum_2)
        
        #print(do_grad)
        #pdb.set_trace()
        if resample_freq is None:
        #    fig_out_name = '%s/output_%s_%s_th%02i_fth%02i_i%04i_j%04i_t%03i_z%03i%s_g%1i_%s'%(      fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,secdataset_proc_figname)
            fig_out_name = '%s/output_%s_%s_th%02ifth%02ii%04ij%04it%03iz%03i%sg%1i_%s'%(      fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,secdataset_proc_figname)
        else:
        #    fig_out_name = '%s/output_%s_%s_th%02i_fth%02i_i%04i_j%04i_t%03i_z%03i%s_g%1i_res%s_%s'%(fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,resample_freq,secdataset_proc_figname)
            fig_out_name = '%s/output_%s_%s_th%02ifth%02ii%04ij%04it%03iz%03i%sg%1ir%s_%s'%(fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,resample_freq,secdataset_proc_figname)
        

        '''
        if fig_lab_d['Dataset 1'] is not None: fig_out_name = fig_out_name + '_d1_%s'%fig_lab_d['Dataset 1']
        if fig_lab_d['Dataset 2'] is not None: fig_out_name = fig_out_name + '_d2_%s'%fig_lab_d['Dataset 2']
        '''
        
        for tmp_datstr in Dataset_lst:
            #if fig_lab_d[tmp_datstr] is not None: 
            fig_out_name = fig_out_name + '_d%s_%s'%(tmp_datstr[-1],fig_lab_d[tmp_datstr])
        



        fig_tit_str_lab = ''
        if load_second_files == False:
            fig_tit_str_lab = fig_lab_d['Dataset 1']
        else:
            if secdataset_proc in Dataset_lst:
                fig_tit_str_lab = '%s'%fig_lab_d[secdataset_proc]
            else:
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]
                if tmpdataset_oper == '-':
                    fig_tit_str_lab = '%s minus %s'%(fig_lab_d[tmpdataset_1],fig_lab_d[tmpdataset_2])
        
        
        

        fig.suptitle( fig_tit_str_lab, fontsize=14)


        fig_out_name = fig_out_name.replace(' ','_')


        if fig_cutout:


            bbox_cutout_pos = [[(but_x1+0.01), (0.066)],[(func_but_x0-0.01),0.965]]
            #bbox_cutout_pos_inches = [[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.066)],[fig.get_figwidth()*(func_but_x0-0.01),fig.get_figheight()*(0.965)]]
            #bbox_cutout_pos_inches = [[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.066)],[fig.get_figwidth()*(func_but_x0-0.01),fig.get_figheight()]]
            bbox_cutout_pos_inches = [[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.066)],[fig.get_figwidth()*(func_but_x0),fig.get_figheight()]]
            bbox_inches =  matplotlib.transforms.Bbox(bbox_cutout_pos_inches)
            
            if verbose_debugging: print('Save Figure: bbox_cutout_pos',bbox_cutout_pos, datetime.now())
            fig.savefig(fig_out_name+ '.png',bbox_inches = bbox_inches, dpi = figdpi)
        else:
            fig.savefig(fig_out_name+ '.png', dpi = figdpi)

        #print('')
        #print(fig_out_name + '.png')
        #print('')





        fig.suptitle(fig_tit_str_int + '\n' + fig_tit_str_lab, fontsize=14)

        try:


            arg_output_text = 'flist1=$(echo "/dir1/file0[4-7]??_*.nc")\n'
            arg_output_text = arg_output_text + 'flist2=$(echo "/dir2/file0[4-7]??_*.nc")\n'
            arg_output_text = arg_output_text + '\n'
            arg_output_text = arg_output_text + "justplot_date_ind='%s'\n"%time_datetime[ti].strftime(date_fmt)
            
            arg_output_text = arg_output_text + '\n\n\n'

            arg_output_text = arg_output_text + 'python NEMO_nc_slevel_viewer.py %s'%configd[1]
            arg_output_text = arg_output_text + ' "$flist1" '
            if zlim_max is not None:arg_output_text = arg_output_text + ' --zlim_max %i'%zlim_max
            arg_output_text = arg_output_text + ' --thin %i'%thd[1]['dx']
            arg_output_text = arg_output_text + ' --thin_files %i'%thd[1]['df']
            arg_output_text = arg_output_text + ' --fig_fname_lab %s'%fig_lab_d['Dataset 1']
            arg_output_text = arg_output_text + ' --lon %f'%lon_d[1][jj,ii]
            arg_output_text = arg_output_text + ' --lat %f'%lat_d[1][jj,ii]
            if cur_xlim is not None: arg_output_text = arg_output_text + ' --xlim %f %f'%(cur_xlim[0],cur_xlim[1])
            if cur_ylim is not None: arg_output_text = arg_output_text + ' --ylim %f %f'%(cur_ylim[0],cur_ylim[1])
            arg_output_text = arg_output_text + ' --date_ind %s'%time_datetime[ti].strftime(date_fmt)
            arg_output_text = arg_output_text + ' --date_fmt %s'%date_fmt
            arg_output_text = arg_output_text + ' --var %s'%var
            arg_output_text = arg_output_text + ' --z_meth %s'%z_meth
            arg_output_text = arg_output_text + ' --zz %s'%zz
            arg_output_text = arg_output_text + ' --do_grad %1i'%do_grad
            arg_output_text = arg_output_text + ' --clim_sym %s'%clim_sym
            #if xlim is not None:arg_output_text = arg_output_text + ' --xlim %f %f'%tuple(xlim)
            #if ylim is not None:arg_output_text = arg_output_text + ' --ylim %f %f'%tuple(ylim)
            if load_second_files:
                #if configd[2] is not None: 
                arg_output_text = arg_output_text + ' --config_2nd %s'%configd[2]
                arg_output_text = arg_output_text + ' --fig_fname_lab_2nd %s'%fig_lab_d['Dataset 2']
                arg_output_text = arg_output_text + ' --thin_2nd %i'%thd[2]['dx']
                arg_output_text = arg_output_text + ' --secdataset_proc "%s"'%secdataset_proc
                arg_output_text = arg_output_text + ' --fname_lst_2nd  "$flist2"'
                arg_output_text = arg_output_text + ' --clim_pair %s'%clim_pair

            arg_output_text = arg_output_text + " --justplot_date_ind '$justplot_date_ind'"
            #arg_output_text = arg_output_text + " --justplot_date_ind '%s'"%time_datetime[ti].strftime(date_fmt)
            arg_output_text = arg_output_text + " --justplot_secdataset_proc '%s'"%justplot_secdataset_proc
            arg_output_text = arg_output_text + " --justplot_z_meth_zz '%s'"%justplot_z_meth_zz
            arg_output_text = arg_output_text + ' --justplot True'       
            arg_output_text = arg_output_text + '\n\n\n'       
            fid = open(fig_out_name + '.txt','w')
            fid.write(arg_output_text)
            fid.close()
            
            print(' ')
            print(fig_out_name + '.png')
            print(fig_out_name + '.txt')
            print(' ')

        except:
            pdb.set_trace()



    ###########################################################################
    # Inner functions defined
    ###########################################################################

    init_timer.append((datetime.now(),'Inner functions created'))

    
    if verbose_debugging: print('Inner functions created ', datetime.now())

    cur_xlim = xlim
    cur_ylim = ylim

    # only load data when needed
    reload_map, reload_ew, reload_ns, reload_hov, reload_ts = True,True,True,True,True


    if justplot: 
        secdataset_proc = just_plt_vals[just_plt_cnt][0]
        #tmp_date_in_ind = just_plt_vals[just_plt_cnt][1]
        z_meth = just_plt_vals[just_plt_cnt][2]
        zz = just_plt_vals[just_plt_cnt][3]



    interp1d_ZwgtT = {}
    interp1d_ZwgtT['Dataset 1'] = {}
    if z_meth_default == 'z_slice':
        interp1d_ZwgtT['Dataset 1'][0] = interp1dmat_create_weight(grid_dict['Dataset 1']['gdept'],0,use_xarray_gdept = use_xarray_gdept)


    for tmp_datstr in Dataset_lst[1:]:
        th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

        interp1d_ZwgtT[tmp_datstr] = {}

        if z_meth_default == 'z_slice':
            if configd[th_d_ind] == configd[1]: #if configd[th_d_ind] is None:
                interp1d_ZwgtT[tmp_datstr] = interp1d_ZwgtT['Dataset 1']
            else:
                interp1d_ZwgtT[tmp_datstr][0] = interp1dmat_create_weight(grid_dict[tmp_datstr]['gdept'],0,use_xarray_gdept = use_xarray_gdept)

    

    if verbose_debugging: print('Interpolation weights created', datetime.now())
    init_timer.append((datetime.now(),'Interpolation weights created'))


    if verbose_debugging: print('Start While Loop', datetime.now())
    #if verbose_debugging: print('')
    #if verbose_debugging: print('')
    if verbose_debugging: print('')

    # initialise button press location
    tmp_press = [(0.5,0.5,)]
    press_ginput = [(0.5,0.5,)]

    hov_y = np.array(0)
    hov_dat_dict, ts_dat_dict = {},{}

    timer_lst = []
    init_timer_lst = []
    if verbose_debugging:
        do_timer = True

    init_timer.append((datetime.now(),'Starting While Loop'))
   
    if verbose_debugging|do_timer|do_memory:
        print()
        
        for i_i in range(1,len(init_timer)):print('Initialisation time %02i - %02i: %s - %s - %s '%(i_i-1,i_i,init_timer[i_i][0] - init_timer[i_i-1][0], init_timer[i_i-1][1], init_timer[i_i][1]))
        print()
    print('Initialisation: total: %s'%(init_timer[-1][0] - init_timer[0][0]))
    if verbose_debugging:print()

    secondary_fig = None

    # if using_set_array
    # paxmap = ax[0].pcolormesh(lon_d[1],lat_d[1],lat_d[1].copy()*np.ma.masked)
    # if using_set_array


    

    while ii is not None:
        # try, exit on error
        if do_timer: timer_lst.append(('Start loop',datetime.now()))
        #pdb.set_trace()
        #if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
        if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

        #process = psutil.Process(os.getpid())
        #print('')
        #print('Memory_use: %f MB'%process.memory_info().rss/1024/1024)  # in bytes 
        #print('')
        #try:
        if True: 
            # extract plotting data (when needed), and subtract off difference files if necessary.

            if verbose_debugging: print('Set current data set (set of nc files) for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if verbose_debugging: print('Convert coordinates for config_2nd', datetime.now())
            #global ii_2nd_ind, jj_2nd_ind, dd_2nd_ind, ew_ii_2nd_ind,ew_jj_2nd_ind,ns_ii_2nd_ind,ns_jj_2nd_ind

            #ii_2nd_ind, jj_2nd_ind = ii,jj
            #ew_ii_2nd_ind, ew_jj_2nd_ind = None, None
            #ns_ii_2nd_ind, ns_jj_2nd_ind = None, None

            ##ew_bl_ii_ind_final,ew_bl_jj_ind_final,ew_wgt = None, None, None
            #ns_bl_ii_ind_final,ns_bl_jj_ind_final,ns_wgt = None, None, None

            iijj_ind = {}
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                #iijj_ind[tmp_datstr] = None
                #if configd[th_d_ind] is not None:
                if configd[th_d_ind] !=  configd[1]:
                    '''
                    if ((configd[1].upper() == 'AMM15') & (configd[th_d_ind].upper() == 'AMM7')) | ((configd[1].upper() == 'AMM7') & (configd[th_d_ind].upper() == 'AMM15')):

                        iijj_ind[tmp_datstr] = {}

                        if ((configd[1].upper() == 'AMM7') & (configd[th_d_ind].upper() == 'AMM15')):

                            lon_mat_rot, lat_mat_rot  = rotated_grid_from_amm15(lon_d[1][jj,ii] ,lat_d[1][jj,ii])
                            ew_lon_mat_rot, ew_lat_mat_rot  = rotated_grid_from_amm15(lon_d[1][jj,:],lat_d[1][jj,:])
                            ns_lon_mat_rot, ns_lat_mat_rot  = rotated_grid_from_amm15(lon_d[1][:,ii],lat_d[1][:,ii])

                            #tmp_lon_arr = lon_rotamm15
                            #tmp_lat_arr = lat_rotamm15
                            tmp_lon_arr = rot_dict[configd[2]]['lon_rot']
                            tmp_lat_arr = rot_dict[configd[2]]['lat_rot']

                            tmp_lon = lon_mat_rot
                            tmp_lat = lat_mat_rot
                            
                            ns_tmp_lon_arr = ns_lon_mat_rot
                            ns_tmp_lat_arr = ns_lat_mat_rot
                            ew_tmp_lon_arr = ew_lon_mat_rot
                            ew_tmp_lat_arr = ew_lat_mat_rot


                        elif ((configd[1].upper() == 'AMM15') & (configd[th_d_ind].upper() == 'AMM7')):

                            tmp_lon_arr = lon
                            tmp_lat_arr = lat

                            tmp_lon = lon_d[1][jj,ii]
                            tmp_lat = lat_d[1][jj,ii]
                            
                            ns_tmp_lon_arr = ns_lon_mat_rot = lon_d[1][:,ii]
                            ns_tmp_lat_arr = ns_lat_mat_rot = lat_d[1][:,ii]
                            ew_tmp_lon_arr = ew_lon_mat_rot = lon_d[1][jj,:]
                            ew_tmp_lat_arr = ew_lat_mat_rot = lat_d[1][jj,:]
                        
                        
                        (iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj'],
                        iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'],
                        iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'], 
                        iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_bl_jj'],
                        iijj_ind[tmp_datstr]['ew_wgt'], 
                        iijj_ind[tmp_datstr]['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_bl_jj'],
                        iijj_ind[tmp_datstr]['ns_wgt']) = regrid_iijj_ew_ns(tmp_lon,tmp_lat,
                            tmp_lon_arr, tmp_lat_arr, 
                            ew_tmp_lon_arr,ew_tmp_lat_arr,
                            ns_tmp_lon_arr,ns_tmp_lat_arr,
                            thd[2]['dx'],thd[2]['y0'],thd[2]['y1'],regrid_meth)

                    else:
                        print('need to code up for differing configs')


                    '''


                    iijj_ind[tmp_datstr] = {}
                    
                    iijj_ind[tmp_datstr]['jj'], iijj_ind[tmp_datstr]['ii'] = regrid_params[tmp_datstr][3][jj,ii],regrid_params[tmp_datstr][4][jj,ii] # ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1][jj,ii],lat_d[1][jj,ii])
                    
                    iijj_ind[tmp_datstr]['ew_jj'],iijj_ind[tmp_datstr]['ew_ii'] = regrid_params[tmp_datstr][3][jj,:],regrid_params[tmp_datstr][4][jj,:]
                    iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_bl_ii'] = regrid_params[tmp_datstr][0][:,jj,:],regrid_params[tmp_datstr][1][:,jj,:]
                    iijj_ind[tmp_datstr]['ew_wgt'] = regrid_params[tmp_datstr][2][:,jj,:]

                    iijj_ind[tmp_datstr]['ns_jj'],iijj_ind[tmp_datstr]['ns_ii'] = regrid_params[tmp_datstr][3][:,ii],regrid_params[tmp_datstr][4][:,ii]
                    iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]['ns_bl_ii'] = regrid_params[tmp_datstr][0][:,:,ii],regrid_params[tmp_datstr][1][:,:,ii]
                    iijj_ind[tmp_datstr]['ns_wgt'] = regrid_params[tmp_datstr][2][:,:,ii]

                    #pdb.set_trace()     
                    '''
                    for ss in iijj_ind[tmp_datstr].keys():ss, iijj_ind[tmp_datstr][ss].shape, type(iijj_ind[tmp_datstr][ss])

                    ('ii', (), <class 'numpy.int64'>)
                    ('jj', (), <class 'numpy.int64'>)
                    ('ew_ii', (486,), <class 'numpy.ma.core.MaskedArray'>)
                    ('ew_jj', (486,), <class 'numpy.ma.core.MaskedArray'>)
                    ('ns_ii', (449,), <class 'numpy.ma.core.MaskedArray'>)
                    ('ns_jj', (449,), <class 'numpy.ma.core.MaskedArray'>)
                    ('ew_bl_ii', (4, 486), <class 'numpy.ndarray'>)
                    ('ew_bl_jj', (4, 486), <class 'numpy.ndarray'>)
                    ('ew_wgt', (4, 486), <class 'numpy.ma.core.MaskedArray'>)
                    ('ns_bl_ii', (4, 449), <class 'numpy.ndarray'>)
                    ('ns_bl_jj', (4, 449), <class 'numpy.ndarray'>)
                    ('ns_wgt', (4, 449), <class 'numpy.ma.core.MaskedArray'>)


                    ######################################

                    iijj_ind[tmp_datstr]['jj'], iijj_ind[tmp_datstr]['ii'] = regrid_params[tmp_datstr][3][jj,ii],regrid_params[tmp_datstr][4][jj,ii] # ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1][jj,ii],lat_d[1][jj,ii])
                    
                    iijj_ind[tmp_datstr]['ew_jj'],iijj_ind[tmp_datstr]['ew_ii'] = regrid_params[tmp_datstr][3][jj,:],regrid_params[tmp_datstr][4][jj,:]
                    iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_bl_ii'] = regrid_params[tmp_datstr][0][:,jj,:],regrid_params[tmp_datstr][1][:,jj,:]
                    iijj_ind[tmp_datstr]['ew_wgt'] = regrid_params[tmp_datstr][2][:,jj,:]

                    iijj_ind[tmp_datstr]['ns_jj'],iijj_ind[tmp_datstr]['ns_ii'] = regrid_params[tmp_datstr][3][:,ii],regrid_params[tmp_datstr][4][:,ii]
                    iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]['ns_bl_ii'] = regrid_params[tmp_datstr][0][:,:,ii],regrid_params[tmp_datstr][1][:,:,ii]
                    iijj_ind[tmp_datstr]['ns_wgt'] = regrid_params[tmp_datstr][2][:,:,ii]


                    ######################################


                    '''



                    '''
                    iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'] = regrid_params[tmp_datstr][4][:,ii],regrid_params[tmp_datstr][3][:,ii]
                    iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'] = regrid_params[tmp_datstr][4][:,ii],regrid_params[tmp_datstr][3][:,ii]


                    iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_bl_jj'] = 0,0
                    iijj_ind[tmp_datstr]['ew_wgt'] = np.ma.array(0)
                    iijj_ind[tmp_datstr]['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_bl_jj'] = 0,0
                    iijj_ind[tmp_datstr]['ns_wgt'] = np.ma.array(0)



                    #ns_tmp_lon_arr = ns_lon_mat_rot = lon_d[1][:,ii]
                    #ns_tmp_lat_arr = ns_lat_mat_rot = lat_d[1][:,ii]
                    #ew_tmp_lon_arr = ew_lon_mat_rot = lon_d[1][jj,:]
                    #ew_tmp_lat_arr = ew_lat_mat_rot = lat_d[1][jj,:]
                    ns_tmp_lon_arr  = lon_d[1][:,ii]
                    ns_tmp_lat_arr  = lat_d[1][:,ii]
                    ew_tmp_lon_arr  = lon_d[1][jj,:]
                    ew_tmp_lat_arr  = lat_d[1][jj,:]
                    pdb.set_trace()
                    iijj_ind[tmp_datstr] = {}
                    print('need to code up for differing configs')
                    iijj_ind[tmp_datstr]['jj'], iijj_ind[tmp_datstr]['ii'] = ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1][jj,ii],lat_d[1][jj,ii])
                    iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_ii'] = ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,ew_tmp_lon_arr,ew_tmp_lat_arr)
                    iijj_ind[tmp_datstr]['ns_jj'], iijj_ind[tmp_datstr]['ns_ii'] = ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,ns_tmp_lon_arr,ns_tmp_lat_arr)
                    pdb.set_trace()
                    '''
                    '''
                    
                            tmp_lon_arr = lon
                            tmp_lat_arr = lat

                            tmp_lon = lon_d[1][jj,ii]
                            tmp_lat = lat_d[1][jj,ii]
                            
                            ns_tmp_lon_arr = ns_lon_mat_rot = lon_d[1][:,ii]
                            ns_tmp_lat_arr = ns_lat_mat_rot = lat_d[1][:,ii]
                            ew_tmp_lon_arr = ew_lon_mat_rot = lon_d[1][jj,:]
                            ew_tmp_lat_arr = ew_lat_mat_rot = lat_d[1][jj,:]
                        

    (iijj_ind[tmp_datstr]['ii'],iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'],iijj_ind[tmp_datstr]['ns_ii'],iijj_ind[tmp_datstr]['ns_jj'], iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_wgt'], iijj_ind[tmp_datstr]['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]['ns_wgt']) = regrid_iijj_ew_ns(tmp_lon,tmp_lat,tmp_lon_arr, tmp_lat_arr, ew_tmp_lon_arr,ew_tmp_lat_arr,ns_tmp_lon_arr,ns_tmp_lat_arr,thd[2]['dx'],thd[2]['y0'],thd[2]['y1'],regrid_meth)








                    plt.plot(iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'],'+-')
                    plt.plot(regrid_params[tmp_datstr][4][jj,:],regrid_params[tmp_datstr][3][jj,:],'x-')
                    plt.show()

                    plt.plot(regrid_params[tmp_datstr][3][iijj_ind[tmp_datstr]['ew_ii'], iijj_ind[tmp_datstr]['ew_jj']],regrid_params[tmp_datstr][4][iijj_ind[tmp_datstr]['ew_ii'], iijj_ind[tmp_datstr]['ew_jj']],'x-')
                  





                    plt.plot(iijj_ind[tmp_datstr]['ew_ii'],iijj_ind[tmp_datstr]['ew_jj'],'+-')
                    plt.plot(regrid_params[tmp_datstr][3][jj,:],regrid_params[tmp_datstr][4][jj,:],'x-')
                    plt.plot(regrid_params[tmp_datstr][3][iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_ii']],regrid_params[tmp_datstr][4][iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_ii']],'x-')
                    plt.plot(regrid_params[tmp_datstr][4][iijj_ind[tmp_datstr]['ew_ii'], iijj_ind[tmp_datstr]['ew_jj']],regrid_params[tmp_datstr][3][iijj_ind[tmp_datstr]['ew_ii'], iijj_ind[tmp_datstr]['ew_jj']],'x-')
                    plt.plot(regrid_params[tmp_datstr][4][iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_ii']],regrid_params[tmp_datstr][3][iijj_ind[tmp_datstr]['ew_jj'], iijj_ind[tmp_datstr]['ew_ii']],'x-')
                    plt.show()

                    iijj_ind[tmp_datstr]['ew_bl_ii'],iijj_ind[tmp_datstr]['ew_bl_jj'] = 0,0
                    iijj_ind[tmp_datstr]['ew_wgt'] = np.ma.array(0)
                    iijj_ind[tmp_datstr]['ns_bl_ii'],iijj_ind[tmp_datstr]['ns_bl_jj'] = 0,0
                    iijj_ind[tmp_datstr]['ns_wgt'] = np.ma.array(0)

                    '''
                        
                

            if verbose_debugging: print('Reload data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if verbose_debugging: print('Reload map, ew, ns, hov, ts',reload_map,reload_ew,reload_ns,reload_hov,reload_ts, datetime.now())
            prevtime = datetime.now()
            datstarttime = prevtime
            

            #to allow the time conversion between file sets with different times
            #if var == 'hs': pdb.set_trace()
            tmp_current_time = time_datetime[ti]
            time_datetime = time_d['Dataset 1'][var_grid['Dataset 1'][var]]['datetime']
            time_datetime_since_1970 = time_d['Dataset 1'][var_grid['Dataset 1'][var]]['datetime_since_1970']
            if nctime_calendar_type in ['360_day','360']:
                #pdb.set_trace()
                ti = np.array([np.abs(ss *360*86400) for ss in (time_datetime - tmp_current_time)]).argmin()
            else:
                ti = np.array([np.abs(ss.total_seconds()) for ss in (time_datetime - tmp_current_time)]).argmin()
            #try:
            #    ti = np.array([np.abs(ss.total_seconds()) for ss in (time_datetime - tmp_current_time)]).argmin()
            #except:
            #    pdb.set_trace()
            ntime = len(time_datetime)
            
            if do_timer: timer_lst.append(('Load Instance',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)


            #### Load data
            ####    (if necess)
            ###################################################################################################
            ###          Preload data
            ###################################################################################################

            if preload_data:
                #print('reload_data_instances:',var,preload_data_var,(data_inst_1 is None),(preload_data_ti != ti),(preload_data_var != var))
                #print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')

                # if data_inst_1 is None (i.e. first loop), or
                #       if the time has changed, or
                #       if the variable has changed
                if  (data_inst is None)|(preload_data_ti != ti)|(preload_data_var != var)|(preload_data_ldi != ldi):

                    data_inst = None

                    if do_memory & do_timer: timer_lst.append(('Deleted data_inst',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    #pdb.set_trace()
                    data_inst,preload_data_ti,preload_data_var,preload_data_ldi= reload_data_instances(var,thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)

                    if do_memory & do_timer: timer_lst.append(('Reloaded data_inst',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                    # For T Diff
                    if do_Tdiff:
                        #data_inst_Tm1['Dataset 1'],data_inst_Tm1['Dataset 2'] = None,None
                        #for tmp_datstr in Dataset_lst:data_inst_Tm1[tmp_datstr] = None
                        data_inst_Tm1 = None
                        data_inst_Tm1 = {}
                        for tmp_datstr in Dataset_lst:data_inst_Tm1[tmp_datstr] = None          
                        preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1 = 0.5,'None',0.5
                        Time_Diff_cnt = 0
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_Tm1',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                if  (data_inst_U is None)|(preload_data_ti_U != ti)|(preload_data_ldi_U != ldi):
                    if vis_curr > 0:


                        data_inst_U = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_U',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_U,preload_data_ti_U,preload_data_var_U,preload_data_ldi_U = reload_data_instances(tmp_var_U,thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_U',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        
                        data_inst_V = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_V,preload_data_ti_V,preload_data_var_V,preload_data_ldi_V = reload_data_instances(tmp_var_V,thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))



                if  (data_inst_mld is None)|(preload_data_ti_mld != ti)|(preload_data_ldi_mld != ldi):
                    if reload_MLD:
                        data_inst_mld = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_mld',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_mld,preload_data_ti_mld,preload_data_var_mld,preload_data_ldi_mld= reload_data_instances(MLD_var,thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)
                        reload_MLD = False
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_mld',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
            ###################################################################################################
            ### Status of buttons
            ###################################################################################################
            
            if do_Tdiff:
                if ti == 0:
                    func_but_text_han['Time Diff'].set_color('0.5')
                else:
                    if Time_Diff:
                        func_but_text_han['Time Diff'].set_color('darkgreen')

                        if (data_inst_Tm1['Dataset 1'] is None)|(preload_data_ti_Tm1 != (ti-1))|(preload_data_var_Tm1 != var)|(preload_data_ldi_Tm1 != ldi):

                            (data_inst_Tm1,preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1) = reload_data_instances(var,thd,ldi,ti-1,
                                    var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)
                            if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_Tm1',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        
                        #pdb.set_trace()
                        if Time_Diff_cnt == 0:
                            #data_inst['Dataset 1'] = data_inst['Dataset 1'] - data_inst_Tm1['Dataset 1']
                            #data_inst['Dataset 2'] = data_inst['Dataset 2'] - data_inst_Tm1['Dataset 2']
                            for tmp_datstr in  Dataset_lst:data_inst[tmp_datstr] = data_inst[tmp_datstr] - data_inst_Tm1[tmp_datstr]
                            Time_Diff_cnt -= 1
                        func_but_text_han['Clim: sym'].set_color('r')
                        #curr_cmap = scnd_cmap
                        clim_sym_but = 1
                        #clim_sym_but_norm_val = clim_sym
                        clim_sym = True

                        reload_map = True
                        reload_ew = True
                        reload_ns = True

                    else:
                        func_but_text_han['Time Diff'].set_color('k')
                        if (data_inst_Tm1['Dataset 1'] is not None):

                            if Time_Diff_cnt == -1:
                                #if (preload_data_ti_Tm1 == (ti-1))|(preload_data_var_Tm1 == var)|(preload_data_ldi_Tm1 == ldi):
                                #data_inst['Dataset 1'] = data_inst['Dataset 1'] + data_inst_Tm1['Dataset 1']
                                #data_inst['Dataset 2'] = data_inst['Dataset 2'] + data_inst_Tm1['Dataset 2']
                                for tmp_datstr in  Dataset_lst:data_inst[tmp_datstr] = data_inst[tmp_datstr] + data_inst_Tm1[tmp_datstr]
                                Time_Diff_cnt += 1

                            func_but_text_han['Clim: sym'].set_color('k')
                            clim_sym_but = 0
                            
                            reload_map = True
                            reload_ew = True
                            reload_ns = True

                            # clear the data_inst_Tm1 array if not in use
                            for tmp_datstr in  Dataset_lst:data_inst_Tm1[tmp_datstr] = None

                if do_memory & do_timer: timer_lst.append(('Applied T_diff',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
                            


            for tmp_datstr in  Dataset_lst:
                if do_mask_dict[tmp_datstr]:
                    if var_dim[var] == 3:
                        data_inst[tmp_datstr] = np.ma.array(data_inst[tmp_datstr], mask = (grid_dict[tmp_datstr]['tmask'][0,:,:] == False))
                    elif var_dim[var] == 4:
                        data_inst[tmp_datstr] = np.ma.array(data_inst[tmp_datstr], mask = (grid_dict[tmp_datstr]['tmask'] == False))


            
            ###################################################################################################
            ### Slice data for plotting 
            ###################################################################################################

            if do_timer: timer_lst.append(('Slice data',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            #pdb.set_trace()
            if reload_map:

                map_dat_dict = None
                
                if do_memory & do_timer: timer_lst.append(('Deleted map_dat_dict',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                
                map_dat_dict = reload_map_data_comb(var,z_meth,zz,zi, data_inst,var_dim, interp1d_ZwgtT,grid_dict,lon_d[1],lat_d[1],regrid_params,regrid_meth,thd,configd,Dataset_lst, use_xarray_gdept = use_xarray_gdept,Sec_regrid = Sec_regrid)
                reload_map = False
                
                if do_memory & do_timer: timer_lst.append(('Reloaded map_dat_dict',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                if vis_curr > 0:
                    reload_UV_map = True

                if do_grad == 1:
                    #pdb.set_trace()
                    
                    for tmp_datstr in  Dataset_lst:

                        ''' if Sec_regrid & (secdataset_proc in Dataset_lst):
                                th_d_ind = int(secdataset_proc[8:])
                                pax.append(ax[0].pcolormesh(lon_d[th_d_ind][::pdy,::pdx],lat_d[th_d_ind][::pdy,::pdx],map_dat[::pdy,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
                            else:
                                pax.append(ax[0].pcolormesh(map_dat_dict['x'][::pdy,::pdx],map_dat_dict['y'][::pdy,::pdx],map_dat[::pdy,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
                            if var_dim[var] == 4:
                        '''
                        #pdb.set_trace()
                        map_dat_dict[tmp_datstr] = field_gradient_2d(map_dat_dict[tmp_datstr], thd[1]['dx']*grid_dict['Dataset 1']['e1t'],thd[1]['dx']*grid_dict['Dataset 1']['e2t'], do_mask = do_mask_dict['Dataset 1'], curr_griddict = grid_dict[tmp_datstr]) # scale up widths between grid boxes
                        
                        # if Sec_regrid is true, do gradient on _Sec_regrid in well, not instead, as needed for clim calcs
                        if Sec_regrid & (tmp_datstr!= 'Dataset 1'):
                            th_d_ind = int(tmp_datstr[8:])
                            #pdb.set_trace()
                            map_dat_dict[tmp_datstr + '_Sec_regrid'] = field_gradient_2d(map_dat_dict[tmp_datstr + '_Sec_regrid'], thd[th_d_ind]['dx']*grid_dict[tmp_datstr]['e1t'],thd[th_d_ind]['dx']*grid_dict[tmp_datstr]['e2t'], do_mask = do_mask_dict[tmp_datstr], curr_griddict = grid_dict[tmp_datstr]) # scale up widths between grid boxes

                    
                    if do_memory & do_timer: timer_lst.append(('Calculated Grad of map_dat_dict',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                    #map_dat_dict['Dataset 1'] = field_gradient_2d(map_dat_dict['Dataset 1'], thd[1]['dx']*grid_dict['Dataset 1']['e1t'],thd[1]['dx']*grid_dict['Dataset 1']['e2t']) # scale up widths between grid boxes
                    #map_dat_dict['Dataset 2'] = field_gradient_2d(map_dat_dict['Dataset 2'], thd[1]['dx']*grid_dict['Dataset 1']['e1t'],thd[1]['dx']*grid_dict['Dataset 1']['e2t']) # map 2 aleady on map1 grid, so use grid_dict['Dataset 1']['e1t'] not grid_dict['Dataset 2']['e1t']
                #pdb.set_trace()

            if verbose_debugging: print('Reloaded map data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()



            if reload_UV_map:
                reload_UV_map = False
                if vis_curr > 0:
                    map_dat_dict_U,map_dat_dict_V = None,None

                    if do_memory & do_timer: timer_lst.append(('Deleted map_dat_dict_U&V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
                    map_dat_dict_U = reload_map_data_comb(tmp_var_U,z_meth,zz,zi, data_inst_U,var_dim, interp1d_ZwgtT,grid_dict,lon_d[1],lat_d[1],regrid_params,regrid_meth,thd,configd,Dataset_lst, use_xarray_gdept = use_xarray_gdept,Sec_regrid = Sec_regrid)
                    map_dat_dict_V = reload_map_data_comb(tmp_var_V,z_meth,zz,zi, data_inst_V,var_dim, interp1d_ZwgtT,grid_dict,lon_d[1],lat_d[1],regrid_params,regrid_meth,thd,configd,Dataset_lst, use_xarray_gdept = use_xarray_gdept,Sec_regrid = Sec_regrid)
              
                    if do_memory & do_timer: timer_lst.append(('Reloaded map_dat_dict_U&V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    


            if verbose_debugging: print('Reloaded vis current map data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            if do_MLD:
                #pdb.set_trace()
                

                if reload_ns:
                    mld_ns_slice_dict = None
                    mld_ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst_mld, lon_d[1], lat_d[1], grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
                if reload_ew:
                    mld_ew_slice_dict = None
                    mld_ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst_mld, lon_d[1], lat_d[1], grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                if do_memory & do_timer: timer_lst.append(('Reloaded MLD ns&ew slices',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
            if reload_ew:
                if var_dim[var] == 4:
                    ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var], regrid_meth,iijj_ind,Dataset_lst,configd)

                    if do_grad == 1:
                        #ew_slice_dict['Dataset 1'], ew_slice_dict['Dataset 2'] = grad_horiz_ew_data(thd,grid_dict,jj, ew_slice_dict['Dataset 1'],ew_slice_dict['Dataset 2'])
                        ew_slice_dict = grad_horiz_ew_data(thd,grid_dict,jj, ew_slice_dict)
                    if do_grad == 2:
                        #ew_slice_dict['Dataset 1'], ew_slice_dict['Dataset 2'] = grad_vert_ew_data(ew_slice_dict['Dataset 1'],ew_slice_dict['Dataset 2'],ew_slice_dict['y'])
                        ew_slice_dict = grad_vert_ew_data(ew_slice_dict)
                else:
                    ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var], regrid_meth,iijj_ind,Dataset_lst,configd)

                reload_ew = False

                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ew_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            

            if verbose_debugging: print('Reloaded  ew data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            if reload_ns:
                if var_dim[var] == 4:               
                    ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                    if do_grad == 1:
                        #ns_slice_dict['Dataset 1'], ns_slice_dict['Dataset 2'] = grad_horiz_ns_data(thd,grid_dict,ii, ns_slice_dict['Dataset 1'],ns_slice_dict['Dataset 2'])
                        ns_slice_dict = grad_horiz_ns_data(thd,grid_dict,ii, ns_slice_dict)
                    if do_grad == 2:
                        #ns_slice_dict['Dataset 1'], ns_slice_dict['Dataset 2'] = grad_vert_ns_data(ns_slice_dict['Dataset 1'],ns_slice_dict['Dataset 2'],ns_slice_dict['y'])
                        ns_slice_dict = grad_vert_ns_data(ns_slice_dict)
                else:

                    ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                  
                reload_ns = False
                
                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ns_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            


            if verbose_debugging: print('Reloaded  ns data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            if profvis:
                #pdb.set_trace()
                pf_dat_dict = reload_pf_data_comb(data_inst,var,var_dim,ii,jj,nz,grid_dict,Dataset_lst,configd,iijj_ind)

                if do_grad == 2:
                    pf_dat_dict = grad_vert_hov_prof_data(pf_dat_dict)


                
                if do_memory & do_timer: timer_lst.append(('Reloaded reload_pf_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            
            if reload_hov:
                if hov_time:
                    if var_dim[var] == 4:
                        #pdb.set_trace()
                        hov_dat_dict = reload_hov_data_comb(var,var_d[1]['mat'],var_grid['Dataset 1'],var_d['d'],ldi,thd, time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict,load_second_files,Dataset_lst,configd)

                        if do_grad == 2:
                            #hov_dat_dict['Dataset 1'],hov_dat_dict['Dataset 2'] = grad_vert_hov_prof_data(hov_dat_dict['Dataset 1'],hov_dat_dict['Dataset 2'],hov_dat_dict['y'])
                            hov_dat_dict = grad_vert_hov_prof_data(hov_dat_dict)
                    else:
                        hov_dat_dict = {}
                        hov_dat_dict['x'] = time_datetime
                        hov_dat_dict['y'] = np.ma.zeros((nz,time_datetime.size))

                else:
                    
                    hov_dat_dict['x'] = time_datetime
                    hov_dat_dict['y'] =  grid_dict['Dataset 1']['gdept'][:,jj,ii]
                    #hov_dat_dict['Dataset 1'] = np.ma.zeros((hov_dat_dict['y'].shape+hov_dat_dict['x'].shape))*np.ma.masked
                    #hov_dat_dict['Dataset 2'] = np.ma.zeros((hov_dat_dict['y'].shape+hov_dat_dict['x'].shape))*np.ma.masked
                    for tmp_datstr in Dataset_lst:hov_dat_dict[tmp_datstr] = np.ma.zeros((hov_dat_dict['y'].shape+hov_dat_dict['x'].shape))*np.ma.masked
                reload_hov = False
                
                if do_memory & do_timer: timer_lst.append(('Reloaded reload_hov_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            

            if verbose_debugging: print('Reloaded hov data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()
            if reload_ts:
                if hov_time:
                    ts_dat_dict = reload_ts_data_comb(var,var_dim,var_grid['Dataset 1'],ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,z_meth,zz,zi,xarr_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,load_second_files)
                else:
                    ts_dat_dict['x'] = time_datetime
                    #ts_dat_dict['Dataset 1'] = np.ma.ones(ntime)*np.ma.masked
                    #ts_dat_dict['Dataset 2'] = np.ma.ones(ntime)*np.ma.masked
                    for tmp_datstr in Dataset_lst:ts_dat_dict[tmp_datstr] = np.ma.ones(ntime)*np.ma.masked
                reload_ts = False

                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ts_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            
            #if Obs and reloading,  
            if do_Obs:
                if reload_Obs:
                    
                    #for a given variable, what obs types to use
                    if var.lower() in ['votemper','votempis','votemper_bot','votempis_bot']:
                        #Obs_var_lst_sub	 = ['ProfT']#,'SST_ins']#,'SST_sat']
                        Obs_var_lst_sub = [ss for ss in Obs_varlst if ss in ['ProfT','SST_ins','SST_sat']]
                    elif var.lower() in ['vosaline']:
                        Obs_var_lst_sub = [ss for ss in Obs_varlst if ss in ['ProfS']]
                    elif var.lower() in ['sossheig']:
                        Obs_var_lst_sub = [ss for ss in Obs_varlst if ss in ['SLA']]
                    elif var.lower() in ['chl']:
                        Obs_var_lst_sub = [ss for ss in Obs_varlst if ss in ['ChlA']]
                    else:
                        Obs_var_lst_sub	 = []
                    
                    # exclude obs types that have been excluded
                    #Obs_var_lst_sub = [ss for ss in Obs_var_lst_sub if ss not in Obs_obstype_hide]
                    #Obs_var_lst_sub = [ss for ss in Obs_var_lst_sub if ss not in Obs_obstype_hide]
                        
                        

                    #pdb.set_trace()
                    Obs_var_lst_sub = [ob_var for ob_var in Obs_var_lst_sub if Obs_vis_d['visible'][ob_var]]



                    print(Obs_var_lst_sub,Obs_vis_d['visible'])



                    #extract relevant day of obs, for each data type, and selected Obs types
                    Obs_dat_dict = {} 
                    for tmp_datstr in Dataset_lst:
                        Obs_dat_dict[tmp_datstr] = {}
                        for ob_var in Obs_var_lst_sub:

                            # If Obs Method is to replace, set the old OPS to zero. 
                            if Obs_reloadmeth == 2:
                                Obs_dict[tmp_datstr][ob_var]['Obs'][ob_ti] = {}

                            # for a give model data time (ti) find nearest Obs data time (ob_ti)
                            Obs_noon_time_minus_current_time =  [(tmpObsdatetime - tmp_current_time).total_seconds() +86400/2 for tmpObsdatetime in  Obs_JULD_datetime_dict[tmp_datstr][ob_var]]
                            ob_ti = np.abs(Obs_noon_time_minus_current_time).argmin()





                            # If Obs Method is to fill or replace, load the current OPS data now.  
                            if (Obs_reloadmeth > 0):

                                tmpObsfname = Obs_fname[tmp_datstr][ob_var][ob_ti]

                                if ob_var in ['ProfT','ProfS']:
                                    Obs_dict[tmp_datstr][ob_var]['Obs'][ob_ti] = load_ops_prof_TS(tmpObsfname,ob_var[-1],excl_qc = True)
                                elif ob_var in ['SST_ins','SST_sat','SLA','ChlA']:
                                    Obs_dict[tmp_datstr][ob_var]['Obs'][ob_ti] = load_ops_2D_xarray(tmpObsfname,ob_var,excl_qc = False)   
                            Obs_dat_dict[tmp_datstr][ob_var] = Obs_dict[tmp_datstr][ob_var]['Obs'][ob_ti]
                    #once reloaded, set to False
                    reload_Obs = False

                    if do_memory & do_timer: timer_lst.append(('Reloaded Obs',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            
            
            if verbose_debugging: print('Reloaded  ts data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            print('Reloaded all data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-datstarttime))

            if do_timer: timer_lst.append(('Data sliced',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)



            
            ###################################################################################################
            ### Check colormaps 
            ###################################################################################################
            
            if verbose_debugging: print('Choose cmap based on secdataset_proc:',secdataset_proc, datetime.now())

            # Choose the colormap depending on which dataset being shown

            if var in ['baroc_phi','barot_phi','VolTran_e3_phi','VolTran_phi']:
                curr_cmap = cylc_cmap
                clim_sym = True
            else:
                if (secdataset_proc in Dataset_lst) & (clim_sym_but != 1):
                    if col_scl == 0:
                        curr_cmap = base_cmap
                    elif col_scl == 1:
                        curr_cmap = base_cmap_high
                    elif col_scl == 2:
                        curr_cmap = base_cmap_low
                    clim_sym = False
                else:
                    curr_cmap = scnd_cmap
                    clim_sym = True
            #else:
            #    print(secdataset_proc)
            #    pdb.set_trace()
            
            ###################################################################################################
            ### Choose which dataset to use
            ###################################################################################################

            pax = []        
            if secdataset_proc in Dataset_lst:
                map_dat = map_dat_dict[secdataset_proc]
                if Sec_regrid & (secdataset_proc!= 'Dataset 1'):
                    map_dat = map_dat_dict[secdataset_proc + '_Sec_regrid']

                if var_dim[var] == 4:
                    #ns_slice_dat = ns_slice_dict[secdataset_proc]
                    #ew_slice_dat = ew_slice_dict[secdataset_proc]
                    hov_dat = hov_dat_dict[secdataset_proc]

                ns_slice_dat = ns_slice_dict[secdataset_proc]
                ew_slice_dat = ew_slice_dict[secdataset_proc]

                ts_dat = ts_dat_dict[secdataset_proc]
                if vis_curr > 0:
                    map_dat_U = map_dat_dict_U[secdataset_proc]
                    map_dat_V = map_dat_dict_V[secdataset_proc]


                if do_MLD:
                    mld_ns_slice_dat = mld_ns_slice_dict[secdataset_proc]
                    mld_ew_slice_dat = mld_ew_slice_dict[secdataset_proc]
            else:
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]
                if tmpdataset_oper == '-':
                    map_dat = map_dat_dict[tmpdataset_1] - map_dat_dict[tmpdataset_2]
                    if var_dim[var] == 4:
                        ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = hov_dat_dict[tmpdataset_1] - hov_dat_dict[tmpdataset_2]

                    elif var_dim[var] == 3:
                        ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]

                    ts_dat = ts_dat_dict[tmpdataset_1] - ts_dat_dict[tmpdataset_2]
                    if vis_curr > 0:
                        map_dat_U = map_dat_dict_U[tmpdataset_1] - map_dat_dict_U[tmpdataset_2]
                        map_dat_V = map_dat_dict_V[tmpdataset_1] - map_dat_dict_V[tmpdataset_2]
                else:
                    pdb.set_trace()

                if do_MLD:  
                    mld_ns_slice_dat = mld_ns_slice_dict['Dataset 1'].copy()*0.
                    mld_ew_slice_dat = mld_ew_slice_dict['Dataset 1'].copy()*0.
                    
            # if in ensemble mode, and an ensmble stat is selected
            # Calculate the stat, and copy into map_dat for the map ax.
            # other vars not encoded. 
            if do_ensemble:

                if var_dim[var] == 4:
                    ns_slice_dat, ew_slice_dat,hov_dat, ens_ts_dat = calc_ens_stat_3d(ns_slice_dat, ew_slice_dat,hov_dat,ns_slice_dict,ew_slice_dict,hov_dat_dict,ts_dat_dict, Ens_stat,Dataset_lst)
                elif var_dim[var] == 3:
                    ens_ns_slice_dat, ens_ew_slice_dat, ens_ts_dat = calc_ens_stat_2d(ns_slice_dict,ew_slice_dict,ts_dat_dict, Ens_stat,Dataset_lst)

                if (Ens_stat is not None):
                    map_dat = calc_ens_stat_map(map_dat_dict, Ens_stat,Dataset_lst)


            ###################################################################################################
            ### Replot data 
            ###################################################################################################

            if do_timer: timer_lst.append(('Plot Data',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)


            mldax_lst = []
            pax2d = []
            if pxy is None:
                pdx, pdy = 1,1
            else:
                if cur_xlim is None:
                    pdx = int(np.ceil(map_dat.shape[1]/pxy))
                else:
                    pdx = int(np.ceil(((ew_slice_dict['x']>cur_xlim[0]) &(ew_slice_dict['x']<cur_xlim[1]) ).sum()/pxy))

                if cur_ylim is None:
                    pdy = int(np.ceil(map_dat.shape[0]/pxy))
                else:
                    pdy = int(np.ceil(((ns_slice_dict['x']>cur_ylim[0]) &(ns_slice_dict['x']<cur_ylim[1]) ).sum()/pxy))
                if pdx <1: pdx = 1
                if pdy <1: pdy = 1
                print('Subsampling pixels (for pxy=%i): pdx = %i; pdy = %i'%(pxy,pdx,pdy))

            if verbose_debugging: print("Do pcolormesh for ii = %i,jj = %i,ti = %i,zz = %i, var = '%s'"%(ii,jj, ti, zz,var), datetime.now())


            # if using_set_array
            #paxmap.set_cmap(curr_cmap)
            #paxmap.set_norm(climnorm)
            #paxmap.set_array(map_dat)  
            ## comment out pax.append(ax[0].pcolormesh(map_dat_dict
            # if using_set_array
            if Sec_regrid & (secdataset_proc in Dataset_lst):
                th_d_ind = int(secdataset_proc[8:])
                pax.append(ax[0].pcolormesh(lon_d[th_d_ind][::pdy,::pdx],lat_d[th_d_ind][::pdy,::pdx],map_dat[::pdy,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
            else:
                pax.append(ax[0].pcolormesh(map_dat_dict['x'][::pdy,::pdx],map_dat_dict['y'][::pdy,::pdx],map_dat[::pdy,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
            if var_dim[var] == 4:
                #pdb.set_trace()
                pax.append(ax[1].pcolormesh(ew_slice_dict['x'][::pdx],ew_slice_dict['y'][:,::pdx],ew_slice_dat[:,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
                pax.append(ax[2].pcolormesh(ns_slice_dict['x'][::pdy],ns_slice_dict['y'][:,::pdy],ns_slice_dat[:,::pdy],cmap = curr_cmap,norm = climnorm, rasterized = True))
                pax.append(ax[3].pcolormesh(hov_dat_dict['x'],hov_dat_dict['y'],hov_dat,cmap = curr_cmap,norm = climnorm, rasterized = True))
            elif var_dim[var] == 3:

                if secdataset_proc in Dataset_lst:
                    
                    for dsi,tmp_datstr in enumerate(Dataset_lst):
                        tmplw = 0.5
                        if secdataset_proc == tmp_datstr:tmplw = 1
                        pax2d.append(ax[1].plot(ew_slice_dict['x'],ew_slice_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))
                        pax2d.append(ax[2].plot(ns_slice_dict['x'],ns_slice_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))

                        if do_ensemble:
                            pax2d.append(ax[1].plot(ew_slice_dict['x'],ens_ew_slice_dat[0],'k', lw = 1))
                            pax2d.append(ax[1].plot(ew_slice_dict['x'],ens_ew_slice_dat[1],'k', lw = 2))
                            pax2d.append(ax[1].plot(ew_slice_dict['x'],ens_ew_slice_dat[2],'k', lw = 1))
                            pax2d.append(ax[2].plot(ns_slice_dict['x'],ens_ns_slice_dat[0],'k', lw = 1))
                            pax2d.append(ax[2].plot(ns_slice_dict['x'],ens_ns_slice_dat[1],'k', lw = 2))
                            pax2d.append(ax[2].plot(ns_slice_dict['x'],ens_ns_slice_dat[2],'k', lw = 1))
                                
                else:
                    # only plot the current dataset difference
                    tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                    tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                    tmpdataset_oper = secdataset_proc[4]
                    if tmpdataset_oper == '-': 
                        

                        pax2d.append(ax[1].plot(ew_slice_dict['x'],ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2],Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[1].plot(ew_slice_dict['x'],ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

                        pax2d.append(ax[2].plot(ns_slice_dict['x'],ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2],Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[2].plot(ns_slice_dict['x'],ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

                        for tmp_datstr1 in Dataset_lst:
                            #th_d_ind1 = int(tmp_datstr1[-1])
                            th_d_ind1 = int(tmp_datstr1[8:])
                            for tmp_datstr2 in Dataset_lst:
                                #th_d_ind2 = int(tmp_datstr2[-1])
                                th_d_ind2 = int(tmp_datstr2[8:])
                                if tmp_datstr1!=tmp_datstr2:
                                    tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)                               
                                    tmplw = 0.5
                                    if secdataset_proc == tmp_diff_str_name:tmplw = 1

                                    pax2d.append(ax[1].plot(ew_slice_dict['x'],ew_slice_dict[tmp_datstr1] - ew_slice_dict[tmp_datstr2],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                    pax2d.append(ax[2].plot(ns_slice_dict['x'],ns_slice_dict[tmp_datstr1] - ns_slice_dict[tmp_datstr2],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                                
                            pax2d.append(ax[1].plot(ew_slice_dict['x'],ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))
                            pax2d.append(ax[2].plot(ns_slice_dict['x'],ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))




            if do_MLD:
                #pdb.set_trace()
                if MLD_show:
                    #pdb.set_trace()
                    if var_dim[var]== 4:
                        mldax_lst.append(ax[1].plot(mld_ew_slice_dict['x'],mld_ew_slice_dat,'k', lw = 0.5))
                        mldax_lst.append(ax[2].plot(mld_ns_slice_dict['x'],mld_ns_slice_dat,'k', lw = 0.5))

            tsax_lst = []
            #Dataset_col = ['r','b','darkgreen','gold']
            # if Dataset X, plot all data sets
            if secdataset_proc in Dataset_lst:
                
                for dsi,tmp_datstr in enumerate(Dataset_lst):
                    tmplw = 0.5
                    if secdataset_proc == tmp_datstr:tmplw = 1
                    tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))
                    if do_ensemble:
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[0],'k', lw = 1))
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[1],'k', lw = 1))
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[2],'k', lw = 1))

                    
            else:
                # only plot the current dataset difference
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]
                if tmpdataset_oper == '-': 
                    
                    tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict[tmpdataset_1] - ts_dat_dict[tmpdataset_2],Dataset_col_diff_dict[secdataset_proc]))
                    tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict['Dataset 1']*0, color = '0.5', ls = '--'))



                    for tmp_datstr1 in Dataset_lst:
                        #th_d_ind1 = int(tmp_datstr1[-1])
                        th_d_ind1 = int(tmp_datstr1[8:])
                        for tmp_datstr2 in Dataset_lst:
                            #th_d_ind2 = int(tmp_datstr2[-1])
                            th_d_ind2 = int(tmp_datstr2[8:])
                            if tmp_datstr1!=tmp_datstr2:
                                tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)                               
                                tmplw = 0.5
                                if secdataset_proc == tmp_diff_str_name:tmplw = 1

                                tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict[tmp_datstr1] - ts_dat_dict[tmp_datstr2],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))

                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict['Dataset 1']*0, color = '0.5', ls = '--'))




                else:
                    pdb.set_trace()


            # if Obs, define some plotting handles
            if do_Obs:
                opax_lst = []
                oxax_lst = []
                opaxtx_lst = []
                #opaxtx = plt.text(None, None, '')

            # Plotting Depth Profiles
            pfax_lst = []
            if profvis:
                pf_xvals = []
                #Dataset_col = ['r','b','darkgreen','gold']
                # if Dataset X, plot all data sets
                if secdataset_proc in Dataset_lst:
                    
                    for dsi,tmp_datstr in enumerate(Dataset_lst):
                        tmplw = 0.5
                        if secdataset_proc == tmp_datstr:tmplw = 1
                        for pfi in pf_dat_dict[tmp_datstr]:pf_xvals.append(pfi)
                        pfax_lst.append(ax[5].plot(pf_dat_dict[tmp_datstr],pf_dat_dict['y'],Dataset_col[dsi], lw = tmplw))

                        # if Obs, plotted the observed data
                        if do_Obs:
                            if Obs_hide == False:
                            
                                # add obs data to pf_xvals to help choose y lims
                                for tmp_datstr in Dataset_lst:
                                    for pfi in obs_obs_sel[tmp_datstr]:pf_xvals.append(pfi)
                                    for pfi in obs_mod_sel[tmp_datstr]:pf_xvals.append(pfi)

                                #plot Obs profile
                                if len(obs_obs_sel[secdataset_proc])==1:
                                    opax_lst.append([ax[5].axvline(obs_obs_sel[secdataset_proc],color = Obs_vis_d['Prof_obs_col'], ls = Obs_vis_d['Prof_obs_ls_2d'], lw = Obs_vis_d['Prof_obs_lw_2d'])])
                                    opax_lst.append([ax[5].axvline(obs_mod_sel[secdataset_proc],color = Obs_vis_d['Prof_mod_col'], ls = Obs_vis_d['Prof_mod_ls_2d'], lw = Obs_vis_d['Prof_mod_lw_2d'])])
                                    
                                else:
                                    
                                    opax_lst.append(ax[5].plot(obs_obs_sel[secdataset_proc], obs_z_sel[secdataset_proc], color=Obs_vis_d['Prof_obs_col'], marker=Obs_vis_d['Prof_obs_ms'], linestyle=Obs_vis_d['Prof_obs_ls'], lw = Obs_vis_d['Prof_obs_lw']))
                                    opax_lst.append(ax[5].plot(obs_mod_sel[secdataset_proc], obs_z_sel[secdataset_proc], color=Obs_vis_d['Prof_mod_col'], marker=Obs_vis_d['Prof_mod_ms'], linestyle=Obs_vis_d['Prof_mod_ls'], lw = Obs_vis_d['Prof_mod_lw']))
                                    #pdb.set_trace()
                                # give id info for Obs
                                if obs_stat_type_sel[tmp_datstr] is not None:
                                    opaxtx_lst.append(ax[5].text(0.02,0.01,'ID = %s\nType: %i\n%s\n%s'%(obs_stat_id_sel[tmp_datstr] ,obs_stat_type_sel[tmp_datstr],obs_stat_time_sel[tmp_datstr].strftime('%Y/%m/%d %H:%M'), lon_lat_to_str(obs_lon_sel[tmp_datstr], obs_lat_sel[tmp_datstr])[0] ), ha = 'left', va = 'bottom', transform=ax[5].transAxes, color = 'k', fontsize = 10,bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none')))
                                    #opaxtx = ax[5].text(0.02,0.01,'ID = %s\nType: %i'%(obs_stat_id_sel[tmp_datstr] ,obs_stat_type_sel[tmp_datstr] ), ha = 'left', va = 'bottom', transform=ax[5].transAxes, color = 'k', fontsize = 10,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none'))

                        
                else:
                    # only plot the current dataset difference
                    tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                    tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                    tmpdataset_oper = secdataset_proc[4]
                    if tmpdataset_oper == '-': 
                        
                        #pf_xvals.append(pf_dat_dict[tmpdataset_1] - pf_dat_dict[tmpdataset_2])
                        for pfi in pf_dat_dict[tmpdataset_1] - pf_dat_dict[tmpdataset_2]:pf_xvals.append(pfi)
                        pfax_lst.append(ax[5].plot(pf_dat_dict[tmpdataset_1] - pf_dat_dict[tmpdataset_2],pf_dat_dict['y'],Dataset_col_diff_dict[secdataset_proc]))
                        pfax_lst.append(ax[5].plot(pf_dat_dict['Dataset 1']*0,pf_dat_dict['y'], color = '0.5', ls = '--'))



                        for tmp_datstr1 in Dataset_lst:
                            #th_d_ind1 = int(tmp_datstr1[-1])
                            th_d_ind1 = int(tmp_datstr1[8:])
                            for tmp_datstr2 in Dataset_lst:
                                #th_d_ind2 = int(tmp_datstr2[-1])
                                th_d_ind2 = int(tmp_datstr2[8:])
                                if tmp_datstr1!=tmp_datstr2:
                                    tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)                               
                                    tmplw = 0.5
                                    if secdataset_proc == tmp_diff_str_name:tmplw = 1

                                    #pf_xvals.append(pf_dat_dict[tmp_datstr1] - pf_dat_dict[tmp_datstr2])
                                    for pfi in pf_dat_dict[tmp_datstr1] - pf_dat_dict[tmp_datstr2]:pf_xvals.append(pfi)
                                    pfax_lst.append(ax[5].plot(pf_dat_dict[tmp_datstr1] - pf_dat_dict[tmp_datstr2],pf_dat_dict['y'],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))

                            pfax_lst.append(ax[5].plot(pf_dat_dict['Dataset 1']*0,pf_dat_dict['y'], color = '0.5', ls = '--'))



                    else:
                        pdb.set_trace()
                #pdb.set_trace()
                pf_xvals_min = np.ma.array(pf_xvals).ravel().min()
                pf_xvals_max = np.ma.array(pf_xvals).ravel().max()
                #pf_xvals_ptp = np.ma.array(pf_xvals).ravel().ptp()
                pf_xvals_ptp = np.ptp(np.ma.array(pf_xvals).ravel())
                #pf_xlim = np.ma.array([np.ma.array(pf_xvals).ravel().min(), np.ma.array(pf_xvals).ravel().max()])
                pf_xlim = np.ma.array([pf_xvals_min-(0.05*pf_xvals_ptp),pf_xvals_max+(0.05*pf_xvals_ptp)])
                #try:
                #    pf_xlim = np.ma.array([np.ma.arra(pf_xvals).ravel().min(), np.ma.concatenate(pf_xvals).ravel().max()])
                #except:
                #    pdb.set_trace()
                if pf_xlim.mask.any():pf_xlim = np.ma.array([0,1])

                #print('pf_xvals limits:',pf_xvals_min,pf_xvals_ptp,pf_xvals_max,pf_xlim)



            
            ###################################################################################################
            ### Title String 
            ###################################################################################################
            
            if do_timer: timer_lst.append(('Data Plotted',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            nice_lev = ''
                
            if z_meth in ['z_slice','z_index']:nice_lev = '%i m'%zz
            elif z_meth == 'ss':nice_lev = 'Surface'
            elif z_meth == 'nb':nice_lev = 'Near-Bed'
            elif z_meth == 'df':nice_lev = 'Surface-Bed'
            elif z_meth == 'zm':nice_lev = 'Depth-Mean'

            if var_dim[var] == 4:  
                map_title_str = '%s (%s); %s %s'%(nice_varname_dict[var],nice_lev,lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0],time_datetime[ti])
            elif var_dim[var] == 3:
                map_title_str = '%s; %s %s'%(nice_varname_dict[var],lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0],time_datetime[ti])

            ax[0].set_title(map_title_str)
            

            ###################################################################################################
            ### add colorbars axes and colorbars
            ###################################################################################################

            if verbose_debugging: print('add colorbars', datetime.now(), 'len(ax):',len(ax))            
            cax = []      
            cbarax = []      
            cbarax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + cbgap, 0.1,cbwid,  0.8]))
            if var_dim[var] == 4:  
                cbarax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap + axwid - cbwid - cbgap + cbgap,0.73, cbwid,  0.17]))
                cbarax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap + axwid - cbwid - cbgap + cbgap,0.52, cbwid,  0.17]))
                cbarax.append(fig.add_axes([leftgap + (axwid - cbwid - cbgap) + wgap + axwid - cbwid - cbgap + cbgap,0.31, cbwid,  0.17]))
            
            
            # if using_set_array
            ## remove 0, so change for ai in [0,1,2,3]: cax.append(plt.colorbar to for ai in [1,2,3]: cax.append(plt.colorbar
            ## add cax.append(plt.colorbar(paxmap, ax = ax[0], cax = cbarax[0]))
            # if using_set_array

            if var_dim[var] == 4:  
                for ai in [0,1,2,3]: cax.append(plt.colorbar(pax[ai], ax = ax[ai], cax = cbarax[ai]))
            elif var_dim[var] == 3:
                for ai in [0]: cax.append(plt.colorbar(pax[ai], ax = ax[ai], cax = cbarax[ai]))
            if verbose_debugging: print('added colorbars', datetime.now(), 'len(ax):',len(ax),'len(cax):',len(cax))
            # apply xlim/ylim if keyword set

            ###################################################################################################
            ### Set x/ylims
            ###################################################################################################


            if cur_xlim is not None:ax[0].set_xlim(cur_xlim)
            if cur_ylim is not None:ax[0].set_ylim(cur_ylim)
            if cur_xlim is not None:ax[1].set_xlim(cur_xlim)
            if cur_ylim is not None:ax[2].set_xlim(cur_ylim)
            if tlim is not None:
                ax[3].set_xlim(tlim)
            else:
                ax[3].set_xlim(ts_dat_dict['x'][[0,-1]])
            if tlim is not None:
                ax[4].set_xlim(tlim)
            else:
                ax[4].set_xlim(ts_dat_dict['x'][[0,-1]])

            #pdb.set_trace()
            #reset ylim to time series to data min max, as long as hovtime as been set once
            #if ax[3].get_xlim() != (0,1.0):
            if var_dim[var] == 4:
                ax[4].set_xlim(ax[3].get_xlim())
            
            if load_second_files == False:
                ax[4].set_ylim(ts_dat.min(),ts_dat.max())
            elif load_second_files:

                '''
                if secdataset_proc == 'Dat1-Dat2':
                    ax[4].set_ylim((ts_dat_dict['Dataset 1'] - ts_dat_dict['Dataset 2']).min(),(ts_dat_dict['Dataset 1'] - ts_dat_dict['Dataset 2']).max())
                elif secdataset_proc == 'Dat2-Dat1':
                    ax[4].set_ylim((ts_dat_dict['Dataset 2'] - ts_dat_dict['Dataset 1']).min(),(ts_dat_dict['Dataset 2'] - ts_dat_dict['Dataset 1']).max())
                '''

                if secdataset_proc in Dataset_lst:
                    tmpts_minmax_lst = []
                    for tmp_datstr in Dataset_lst:tmpts_minmax_lst.append(ts_dat_dict[tmp_datstr].min())
                    for tmp_datstr in Dataset_lst:tmpts_minmax_lst.append(ts_dat_dict[tmp_datstr].max())
                    ax[4].set_ylim(np.ma.array(tmpts_minmax_lst).min(),np.ma.array(tmpts_minmax_lst).max())
                    del(tmpts_minmax_lst)

                else:
                    tmpts_minmax_lst = []
                    for tmp_datstr1 in Dataset_lst:
                        #th_d_ind1 = int(tmp_datstr1[-1])
                        th_d_ind1 = int(tmp_datstr1[8:])
                        for tmp_datstr2 in Dataset_lst:
                            #th_d_ind2 = int(tmp_datstr2[-1])
                            th_d_ind2 = int(tmp_datstr2[8:])
                            if tmp_datstr1!=tmp_datstr2:
                                tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)                               

                                for tmp_datstr in Dataset_lst:tmpts_minmax_lst.append((ts_dat_dict[tmp_datstr1] - ts_dat_dict[tmp_datstr2]).min())
                                for tmp_datstr in Dataset_lst:tmpts_minmax_lst.append((ts_dat_dict[tmp_datstr1] - ts_dat_dict[tmp_datstr2]).max())


                    ax[4].set_ylim(np.ma.array(tmpts_minmax_lst).min(),np.ma.array(tmpts_minmax_lst).max())
                    del(tmpts_minmax_lst)
                    del(tmp_diff_str_name)

            if verbose_debugging: print('Set x y lims', datetime.now())

            # set minimum depth if keyword set
            zlim_min = 1
            zlim_min = -0.5
            if zlim_max == None:
                tmpew_xlim = ax[1].get_xlim()
                tmpns_xlim = ax[2].get_xlim()
                tmpew_visible_ind = (ew_slice_dict['x']>=tmpew_xlim[0]) & (ew_slice_dict['x']<=tmpew_xlim[1]) 
                tmpns_visible_ind = (ns_slice_dict['x']>=tmpns_xlim[0]) & (ns_slice_dict['x']<=tmpns_xlim[1]) 

                tmp_ew_ylim = [0,zlim_min]
                tmp_ns_ylim = [0,zlim_min]
                if tmpew_visible_ind.any(): tmp_ew_ylim = [ew_slice_dict['y'][:,tmpew_visible_ind].max(),zlim_min]
                if tmpns_visible_ind.any(): tmp_ns_ylim = [ns_slice_dict['y'][:,tmpns_visible_ind].max(),zlim_min]
                tmp_hov_ylim = [hov_dat_dict['y'].max(),zlim_min]
                if var_dim[var] == 4:
                    ax[1].set_ylim(tmp_ew_ylim)
                    ax[2].set_ylim(tmp_ns_ylim)
                ax[3].set_ylim(tmp_hov_ylim)

                if profvis:
                    tmp_py_ylim = [pf_dat_dict['y'].max(),zlim_min]
                    ax[5].set_ylim(tmp_py_ylim)
                    ax[5].set_xlim(pf_xlim)
                    #
            else:
                if var_dim[var] == 4:
                    ax[1].set_ylim([zlim_max,zlim_min])
                    ax[2].set_ylim([zlim_max,zlim_min])
                ax[3].set_ylim([np.minimum(zlim_max,hov_dat_dict['y'].max()),zlim_min])
                if profvis:
                    tmp_py_ylim = [np.minimum(zlim_max,pf_dat_dict['y'].max()),zlim_min]
                    ax[5].set_ylim(tmp_py_ylim)
                    ax[5].set_xlim(pf_xlim)
                #pdb.set_trace()

            if var_dim[var] == 3:
                tmpew_xlim = ax[1].get_xlim()
                tmpns_xlim = ax[2].get_xlim()
                tmpew_visible_ind = (ew_slice_dict['x']>=tmpew_xlim[0]) & (ew_slice_dict['x']<=tmpew_xlim[1]) 
                tmpns_visible_ind = (ns_slice_dict['x']>=tmpns_xlim[0]) & (ns_slice_dict['x']<=tmpns_xlim[1]) 
                # catch edgecase where cross hairs don't pass through any water
                tmp_ew_slice_subset = ew_slice_dat[tmpew_visible_ind]
                tmp_ns_slice_subset = ns_slice_dat[tmpns_visible_ind]
                if (tmp_ew_slice_subset.size>0)&(not tmp_ew_slice_subset.mask.all()):
                    #tmp_ew_ylim = np.array([ew_slice_dat[tmpew_visible_ind].min(),ew_slice_dat[tmpew_visible_ind].max()])
                    tmp_ew_ylim = np.array([tmp_ew_slice_subset.min(),tmp_ew_slice_subset.max()])
                    ax[1].set_ylim(tmp_ew_ylim)
                    #try:
                    #    ax[1].set_ylim(tmp_ew_ylim)
                    #except:
                    #    pdb.set_trace()
                if (tmp_ns_slice_subset.size>0)&(not tmp_ns_slice_subset.mask.all()):
                    #tmp_ns_ylim = np.array([ns_slice_dat[tmpns_visible_ind].min(),ns_slice_dat[tmpns_visible_ind].max()])
                    tmp_ns_ylim = np.array([tmp_ns_slice_subset.min(),tmp_ns_slice_subset.max()])
                    ax[2].set_ylim(tmp_ns_ylim)
                    #try: 
                    #    ax[2].set_ylim(tmp_ns_ylim)
                    #except: 
                    #    pdb.set_trace()
                        
                del(tmp_ew_slice_subset)
                del(tmp_ns_slice_subset)
        
            ###################################################################################################
            ### add color lims
            ###################################################################################################

            if do_timer: timer_lst.append(('Starting clim',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            tmpxlim = cur_xlim
            tmpylim = cur_ylim
            if cur_xlim is None: tmpxlim = ax[0].get_xlim()#np.array([lon_d[1].min(), lon_d[1].max()])    
            if cur_ylim is None: tmpylim = ax[0].get_ylim()#np.array([lat_d[1].min(), lat_d[1].max()])  

            if verbose_debugging: print('Reset colour limits', datetime.now())
            try:

                if load_second_files & (clim_pair == True)&(secdataset_proc  in Dataset_lst) :

                    '''
                    # if no xlim present using those from the map.
                    tmpxlim = cur_xlim
                    tmpylim = cur_ylim
                    if cur_xlim is None: tmpxlim = ax[0].get_xlim()#np.array([lon_d[1].min(), lon_d[1].max()])    
                    if cur_ylim is None: tmpylim = ax[0].get_ylim()#np.array([lat_d[1].min(), lat_d[1].max()])    
                    '''

                    map_dat_reg_mask_1 = (lon_d[1]>tmpxlim[0]) & (lon_d[1]<tmpxlim[1]) & (lat_d[1]>tmpylim[0]) & (lat_d[1]<tmpylim[1])

                    tmp_map_dat_clim_lst = []
                    for tmp_datstr in Dataset_lst:
                        
                        tmp_map_dat_clim = map_dat_dict[tmp_datstr][map_dat_reg_mask_1]
                        tmp_map_dat_clim = tmp_map_dat_clim[tmp_map_dat_clim.mask == False]

                        if len(tmp_map_dat_clim)>2:
                            tmp_map_dat_clim_lst.append(np.percentile(tmp_map_dat_clim,(5,95)))
                        
                        
                    tmp_map_dat_clim_mat = np.ma.array(tmp_map_dat_clim_lst).ravel()
                    if tmp_map_dat_clim_mat.size>1:
                        map_clim = np.ma.array([tmp_map_dat_clim_mat.min(),tmp_map_dat_clim_mat.max()])

                        if clim_sym: map_clim = np.ma.array([-1,1])*np.abs(map_clim).max()
                        if map_clim.mask.any() == False: set_clim_pcolor(map_clim, ax = ax[0])

                    
                    # only apply to ns and ew slices, and hov if 3d variable. 

                    if var_dim[var] == 4:

                        ew_dat_reg_mask_1 = (ew_slice_dict['x']>tmpxlim[0]) & (ew_slice_dict['x']<tmpxlim[1]) 
                        ns_dat_reg_mask_1 = (ns_slice_dict['x']>tmpylim[0]) & (ns_slice_dict['x']<tmpylim[1])
                        
                        tmp_ew_dat_clim_lst,tmp_ns_dat_clim_lst, tmp_hov_dat_clim_lst = [],[],[]

                        for tmp_datstr in Dataset_lst:

                            tmp_ew_dat_clim = ew_slice_dict[tmp_datstr][:,ew_dat_reg_mask_1]
                            tmp_ns_dat_clim = ns_slice_dict[tmp_datstr][:,ns_dat_reg_mask_1]
                            tmp_hov_dat_clim = hov_dat_dict[tmp_datstr].copy()

                            tmp_ew_dat_clim = tmp_ew_dat_clim[tmp_ew_dat_clim.mask == False]
                            tmp_ns_dat_clim = tmp_ns_dat_clim[tmp_ns_dat_clim.mask == False]
                            tmp_hov_dat_clim = tmp_hov_dat_clim[tmp_hov_dat_clim.mask == False]


                            if len(tmp_ew_dat_clim)>2:   
                                tmp_ew_dat_clim_lst.append(np.percentile(tmp_ew_dat_clim,(5,95)))

                            if len(tmp_ns_dat_clim)>2:   
                                tmp_ns_dat_clim_lst.append(np.percentile(tmp_ns_dat_clim,(5,95)))

                            if len(tmp_hov_dat_clim)>2:  
                                tmp_hov_dat_clim_lst.append(np.percentile(tmp_hov_dat_clim,(5,95)))




                        tmp_ew_dat_clim_mat =  np.ma.array(tmp_ew_dat_clim_lst).ravel()
                        tmp_ns_dat_clim_mat =  np.ma.array(tmp_ns_dat_clim_lst).ravel()
                        tmp_hov_dat_clim_mat = np.ma.array(tmp_hov_dat_clim_lst).ravel()


                        if tmp_ew_dat_clim_mat.size>1:
                            ew_clim = np.ma.array([tmp_ew_dat_clim_mat.min(),tmp_ew_dat_clim_mat.max()])
                            if clim_sym: ew_clim = np.ma.array([-1,1])*np.abs(ew_clim).max()
                            if ew_clim.mask.any() == False: set_clim_pcolor(ew_clim, ax = ax[1])

                        if tmp_ns_dat_clim_mat.size>1:
                            ns_clim = np.ma.array([tmp_ns_dat_clim_mat.min(),tmp_ns_dat_clim_mat.max()])
                            if clim_sym: ns_clim = np.ma.array([-1,1])*np.abs(ns_clim).max()
                            if ns_clim.mask.any() == False: set_clim_pcolor(ns_clim, ax = ax[2])

                        if tmp_hov_dat_clim_mat.size>1:
                            hov_clim = np.ma.array([tmp_hov_dat_clim_mat.min(),tmp_hov_dat_clim_mat.max()])
                            if clim_sym: hov_clim = np.ma.array([-1,1])*np.abs(hov_clim).max()
                            if hov_clim.mask.any() == False: set_clim_pcolor(hov_clim, ax = ax[3])
#
                            
                else:
                    if (clim is None)| (secdataset_proc not in Dataset_lst):
                        for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)
                        
                    elif clim is not None:
                        if len(clim)>2:
                            for ai,tmpax in enumerate(ax):set_clim_pcolor((clim[2*ai:2*ai+1+1]), ax = tmpax)
                            set_clim_pcolor((clim[:2]), ax = ax[0])
                        elif len(clim)==2:
                            for ai,tmpax in enumerate(ax):set_clim_pcolor((clim), ax = tmpax)
                            set_clim_pcolor((clim), ax = ax[0])
            except:
                print("An exception occured - probably 'IndexError: cannot do a non-empty take from an empty axes.'")
                pdb.set_trace()

            if do_timer: timer_lst.append(('Set clim',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

    
            ###################################################################################################
            ### add current loc lines
            ###################################################################################################

            if do_timer: timer_lst.append(('Add current location lines',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            if verbose_debugging: print('Plot location lines for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            
            ## add lines to show current point. 
            # using plot for the map to show lines if on a rotated grid (amm15) etc.
            cs_plot_1 = ax[0].plot(lon_d[1][jj,:],lat_d[1][jj,:],color = '0.5', alpha = 0.5) 
            cs_plot_2 = ax[0].plot(lon_d[1][:,ii],lat_d[1][:,ii],color = '0.5', alpha = 0.5)
            cs_line = []
            # using axhline, axvline, for slices, hov, time series
            cs_line.append(ax[1].axvline(lon_d[1][jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[2].axvline(lat_d[1][jj,ii],color = '0.5', alpha = 0.5))
            cs_line.append(ax[3].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))
            cs_line.append(ax[4].axvline(time_datetime_since_1970[ti],color = '0.5', alpha = 0.5))
            if var_dim[var] == 4:
                cs_line.append(ax[1].axhline(zz,color = '0.5', alpha = 0.5))
                cs_line.append(ax[2].axhline(zz,color = '0.5', alpha = 0.5))
            cs_line.append(ax[3].axhline(zz,color = '0.5', alpha = 0.5))
            if np.prod(ax[4].get_ylim())<0: # if xlim straddles zero
                cs_line.append(ax[4].axhline(0,color = '0.5', alpha = 0.5))
                
            if profvis:
                cs_line.append(ax[5].axhline(zz,color = '0.5', alpha = 0.5))
                if np.prod(ax[5].get_xlim())<0: # if ylim straddles zero
                    cs_line.append(ax[5].axvline(0,color = '0.5', alpha = 0.5))
                    

            
            ###################################################################################################
            ### add dataset labels
            ###################################################################################################

            #if fig_lab_d['Dataset 1']: tsaxtx1.set_text(fig_lab_d['Dataset 1'])

            if load_second_files:       
                if secdataset_proc in Dataset_lst:
                    #for tsaxtx in tsaxtx_lst:tsaxtx.set_visible(True)
                    for tsaxtxd in tsaxtxd_lst:tsaxtxd.set_visible(False)
                else:
                    #for tsaxtx in tsaxtx_lst:tsaxtx.set_visible(False)
                    for tsaxtxd in tsaxtxd_lst:tsaxtxd.set_visible(True)

               
            ###################################################################################################
            ### add contours
            ###################################################################################################
            if do_timer: timer_lst.append(('Do Contours',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            conax = [] # define it outside if statement
            if do_cont:


                contcols, contlws, contalphas = '0.5',0.5,0.5
                cont_val_lst = []
                
                for tmpcax in cax:cont_val_lst.append(get_colorbar_values(tmpcax))
                
                #conax.append(ax[0].contour(map_dat_dict['x'],map_dat_dict['y'],map_dat,cont_val_lst[0], colors = contcols, linewidths = contlws, alphas = contalphas))
                                
                if Sec_regrid & (secdataset_proc in Dataset_lst):
                    th_d_ind = int(secdataset_proc[8:])
                    conax.append(ax[0].contour(lon_d[th_d_ind],lat_d[th_d_ind],map_dat,cont_val_lst[0], colors = contcols, linewidths = contlws, alphas = contalphas))
                else:
                    conax.append(ax[0].contour(map_dat_dict['x'],map_dat_dict['y'],map_dat,cont_val_lst[0], colors = contcols, linewidths = contlws, alphas = contalphas))
                 
                
                
                if var_dim[var] == 4: 
                    conax.append(ax[1].contour(np.tile(ew_slice_dict['x'],(nz,1)),ew_slice_dict['y'],ew_slice_dat,cont_val_lst[1], colors = contcols, linewidths = contlws, alphas = contalphas))
                    conax.append(ax[2].contour(np.tile(ns_slice_dict['x'],(nz,1)),ns_slice_dict['y'],ns_slice_dat,cont_val_lst[2], colors = contcols, linewidths = contlws, alphas = contalphas))
                    if hov_time & ntime>1:
                        conax.append(ax[3].contour(hov_dat_dict['x'],hov_dat_dict['y'],hov_dat,cont_val_lst[3], colors = contcols, linewidths = contlws, alphas = contalphas))

            
            
            ###################################################################################################
            ### add Observations to map as scatter plot to 
            ###################################################################################################
            
            if do_timer: timer_lst.append(('Do Observations',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            if do_Obs:

                oax_lst = []
                tmp_obs_lst,tmp_obs_llind_lst = [],[]
                # if selected a dataset, rather than the difference between datasets
                if secdataset_proc in Dataset_lst:
                    # get current colorlimits
                    obs_clim = get_clim_pcolor(ax = ax[0])

                    
                    # for each Obs obs type, 
                    for obvi,ob_var in enumerate(Obs_var_lst_sub):
                        #extract the lon, lat, z and OBS
                        #pdb.set_trace()
                        tmpobsx = Obs_dat_dict[secdataset_proc][ob_var]['LONGITUDE']
                        tmpobsy = Obs_dat_dict[secdataset_proc][ob_var]['LATITUDE']
                        tmpobsz = Obs_dat_dict[secdataset_proc][ob_var]['DEPTH']
                        if Obs_AbsAnom:
                            tmpobsdat_mat = Obs_dat_dict[secdataset_proc][ob_var]['OBS']
                        else:
                            tmpobsdat_mat = Obs_dat_dict[secdataset_proc][ob_var]['MOD_HX'] - Obs_dat_dict[secdataset_proc][ob_var]['OBS']

                        Obs_scatSS = Obs_vis_d['Scat_symsize'][ob_var] 
                        if Obs_hide_edges:
                            Obs_scatEC = None
                        else:
                            Obs_scatEC = Obs_vis_d['Scat_edgecol'][ob_var] 


                        # if Obs variable is a 2d var
                        if len(tmpobsdat_mat.shape) == 1:

                            tmpobsz = tmpobsz.reshape(-1,1)
                            tmpobsdat_mat = tmpobsdat_mat.reshape(-1,1)

                        # choose the obs to plot, depending on the current depth (surface, zslice etc.)
                        if z_meth in ['z_slice','ss','z_index']:
                            if z_meth == 'z_slice':  obs_tmp_zz = zz
                            if z_meth == 'z_index': 
                                obs_tmp_zz = zz
                                print('check ops for z_index')
                            # find the obs nearst to depth zi, or surface
                            obs_obs_zi_lst = np.ma.abs(tmpobsz - zz).argmin(axis = 1)
                            tmpobsdat = np.ma.array([tmpobsdat_mat[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                        elif z_meth == 'nb':
                            # find deepest obs
                            obs_obs_zi_lst = tmpobsz.argmax(axis = 1)
                            tmpobsdat = np.ma.array([tmpobsdat_mat[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                        elif z_meth == 'df':
                            # find deepest and shallowst obs for df
                            obs_obs_zi_lst = np.ma.abs(tmpobsz - 0).argmin(axis = 1)
                            tmpobsdat_ss = np.ma.array([tmpobsdat_mat[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                            obs_obs_zi_lst = tmpobsz.argmax(axis = 1)
                            tmpobsdat_nb = np.ma.array([tmpobsdat_mat[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                            tmpobsdat = tmpobsdat_ss - tmpobsdat_nb
                        elif z_meth == 'zm':
                            #depth mean obs if zm.
                            tmpobsdat = tmpobsdat_mat.mean(axis = 1)
                        else:

                            pdb.set_trace()
                        #pdb.set_trace()
                        tmpobslatlonind = (tmpobsx>tmpxlim[0]) & (tmpobsx<tmpxlim[1]) & (tmpobsy>tmpylim[0]) & (tmpobsy<tmpylim[1]) 
                        tmp_obs_lst.append(tmpobsdat)
                        tmp_obs_llind_lst.append(tmpobslatlonind)
                        #scatter plot them
                        if Obs_hide == False:
                            if Obs_AbsAnom:
                                oax_lst.append(ax[0].scatter(tmpobsx,tmpobsy,c = tmpobsdat, vmin = obs_clim[0],vmax = obs_clim[1], s = Obs_scatSS, edgecolors = Obs_scatEC ))
                            else:

                                oax_lst.append(ax[0].scatter(tmpobsx,tmpobsy,c = tmpobsdat, s = Obs_scatSS, edgecolors = Obs_scatEC, cmap = matplotlib.cm.seismic ))
                                

                                '''
                                if obvi == 0:
                                    obs_OmB_clim = np.percentile(np.abs(tmpobsdat[tmpobsdat.mask == False]),95)*np.array([-1,1])    
                                oax_lst.append(ax[0].scatter(tmpobsx,tmpobsy,c = tmpobsdat, vmin = obs_OmB_clim[0],vmax = obs_OmB_clim[1], s = Obs_scatSS, edgecolors = Obs_scatEC, cmap = matplotlib.cm.seismic ))

                                if obvi == 0:
                                    #cbarax.append(fig.add_axes([0.3,0.13, 0.15,cbwid]))
                                    #cax.append(plt.colorbar(oax_lst[0], ax = ax[0], cax = cbarax[-1], orientation = 'horizontal'))
                                    cbarobsax = [fig.add_axes([0.305,0.11, 0.15,0.02])]
                                    #cbarobsax[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                                    cax.append(plt.colorbar(oax_lst[0], ax = ax[0], cax = cbarobsax[0], orientation = 'horizontal'))
                                    cax[-1].ax.xaxis.set_ticks_position('top')
                                    cax[-1].ax.xaxis.set_label_position('top')
                                '''

                    if (len(tmp_obs_lst)>0)& (Obs_AbsAnom==False) & (Obs_hide ==  False):
                        #try:

                        tmp_obs_mat=np.ma.concatenate(tmp_obs_lst)
                        tmp_obs_llind_mat=np.ma.concatenate(tmp_obs_llind_lst)
                        if tmp_obs_llind_mat.sum()>3:
                            obs_OmB_clim = np.percentile(np.abs(tmp_obs_mat[(tmp_obs_mat.mask == False) & tmp_obs_llind_mat]),95)*np.array([-1,1])    
                        else:
                            obs_OmB_clim = np.percentile(np.abs(tmp_obs_mat[tmp_obs_mat.mask == False]),95)*np.array([-1,1])    
                        for tmp_oax_lst in oax_lst: tmp_oax_lst.set_clim(obs_OmB_clim)          
                        cbarobsax = [fig.add_axes([0.305,0.11, 0.15,0.02])]
                        #cbarobsax[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                        cax.append(plt.colorbar(oax_lst[0], ax = ax[0], cax = cbarobsax[0], orientation = 'horizontal'))
                        cax[-1].ax.xaxis.set_ticks_position('top')
                        cax[-1].ax.xaxis.set_label_position('top')
                        del(tmp_obs_lst)
                        del(tmp_obs_mat)
                        del(tmp_obs_llind_mat)
                        del(tmp_obs_llind_lst)
                        #except:
                         #   pdb.set_trace()
                                
                    
                    # plt selected obs marker
                            
                    if Obs_hide == False:
                        oxax_lst.append(ax[0].plot(obs_lon_sel[secdataset_proc],obs_lat_sel[secdataset_proc], 'kx', ms = 12))
                    
                    #re apply xlim and ylim to cut out scatter outside domain. 
                    ax[0].set_xlim(tmpxlim)
                    ax[0].set_ylim(tmpylim)

            ###################################################################################################
            ### add vectors
            ###################################################################################################
            if do_timer: timer_lst.append(('Do Vectors',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            visax = []
            if vis_curr > 0:  
                if vis_curr_meth == 'barb':

                    vis_barb_per_side = 35# 25,50

                    # Sqrt of how many data points in the current map axis
                    vis_pnts_vis = np.sqrt(get_pnts_pcolor_in_region(ax = ax[0]))

                    # Sqrt of product of axes range (degrees lon * degrees lat) in the current map axis
                    #vis_xylim_vis=np.sqrt(tmpxlim.ptp()*tmpylim.ptp())
                    vis_xylim_vis=np.sqrt(np.ptp(tmpxlim)*np.ptp(tmpylim))

                    # how many U/V points to skip to give ~vis_barb_per_side (50) current barbs per side.
                    vis_ev = int(   np.maximum(   vis_pnts_vis//vis_barb_per_side,   1)   )

                    #legnth of current barbs (product of fixed lenght and scale factor.)
                    vis_fix_scf = 0.8*vis_xylim_vis/(vis_pnts_vis/(vis_ev))

                    # given a scale factor of 4, what is the fixed length.
                    #vis_fixed_len = 0.05
                    vis_scf = 4
                    vis_fixed_len = vis_fix_scf/vis_scf

                    # To make current barbs clear on dark and light background, either:
                    #       add a dotted white barb over a black one or 
                    #visax.append(current_barb(map_dat_dict['x'],map_dat_dict['y'],map_dat_U,map_dat_V,fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev, color = 'k', ax = ax[0], linewidth = 0.4))
                    #visax.append(current_barb(map_dat_dict['x'],map_dat_dict['y'],map_dat_U,map_dat_V,fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev, color = 'w', ax = ax[0], linewidth = 0.4, linestyle = 'dotted'))
                    #       give a black outline to a white barb,
                    #visax.append(current_barb(map_dat_dict['x'],map_dat_dict['y'],map_dat_U,map_dat_V,fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev, color = 'k', ax = ax[0], linewidth = 2,))
                    #visax.append(current_barb(map_dat_dict['x'],map_dat_dict['y'],map_dat_U,map_dat_V,fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev, color = 'w', ax = ax[0], linewidth = 0.5))
                    #       or use path effects

                    vis_pe = [pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]

                    visax.append(current_barb(map_dat_dict['x'][::pdy,::pdx],map_dat_dict['y'][::pdy,::pdx],map_dat_U[::pdy,::pdx],map_dat_V[::pdy,::pdx],
                                              fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev,ax = ax[0], 
                                              color = 'w',linewidth=0.75, path_effects=vis_pe))
                    
                    ''' 
                    
                                        
                    fig2 = plt.figure()
                    tmpax2 = plt.subplot(111)
                    plt.pcolormesh(map_dat_dict['x'],map_dat_dict['y'],np.sqrt(map_dat_U**2 + map_dat_V**2))
                    plt.xlim(tmpxlim)
                    plt.ylim(tmpylim)
                    vis_pe = [pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
                    current_barb(map_dat_dict['x'],map_dat_dict['y'],map_dat_U,map_dat_V, color = 'w',linewidth=0.75, path_effects=vis_pe,fixed_len = vis_fixed_len,scf = vis_scf,evx = vis_ev,evy = vis_ev,ax = tmpax2)
                    fig2.show()
                    fig2.close()

                    '''
                    
                    #pdb.set_trace()

                    """
                    vis_ev = 1



                    # find a mask of points currently displayed.
                    map_dat_reg_mask_UV = (lon_d[1]>tmpxlim[0]) & (lon_d[1]<tmpxlim[1]) & (lat_d[1]>tmpylim[0]) & (lat_d[1]<tmpylim[1])

                    # count the points, and square root them (to give an idea of number of points along each side),
                    # and divide by 100, so give ~100 vectors along each side... then make sure greater than 1, and an integer
                    
                    # Root of how many points are visible in the current map - an idea of the points along each side
                    vis_pnts_vis=np.sqrt(map_dat_reg_mask_UV.sum())

                    # Root of the product of the xlim and ylim of the current map - an idea of the degrees along each side
                    vis_xylim_vis=np.sqrt(tmpxlim.ptp()*tmpylim.ptp())

                    # how many U and V points skipped.
                    vis_ev = int(   np.maximum(   vis_pnts_vis//100,   1)   )
                    
                    # reduce input x, y, U and V.
                    vis_x = map_dat_dict['x'][::vis_ev,::vis_ev]
                    vis_y = map_dat_dict['y'][::vis_ev,::vis_ev]
                    vis_U = map_dat_U[::vis_ev,::vis_ev]
                    vis_V = map_dat_V[::vis_ev,::vis_ev]

                    # Root of how many U and V points in cuurent map
                    vis_UV_pnts_vis = np.sqrt(map_dat_reg_mask_UV[::vis_ev,::vis_ev].sum())

                    # product of scale factor and fixed lenght = degree each side/UV points on each side * 1.25
                    vis_fix_scf = 1.25*vis_xylim_vis/vis_UV_pnts_vis

                    #vis_fix_scf = vis_xylim_vis/vis_ev/0.8/5

                    vis_fixed_len = 0.05
                    vis_scf = 4

                    vis_fixed_len = vis_fix_scf/vis_scf
                    '''
                    print('vis_curr vis_ev = ',vis_ev)
                    print('vis_curr vis_fixed_len = ',vis_fixed_len)
                    print('vis_curr vis_pnts_vis = ',vis_pnts_vis)
                    print('vis_curr vis_xylim_vis = ',vis_xylim_vis)
                    print('vis_curr vis_UV_pnts_vis = ',vis_UV_pnts_vis)
                    print('vis_curr vis_fix_scf = ',vis_fix_scf)
                    '''
                    if vis_ev <1: pdb.set_trace()

                    vis_offset = 0.#np.sqrt((vis_x[1:,1:]-vis_x[:-1,:-1])**2 + (vis_y[1:,1:]-vis_y[:-1,:-1])**2 ).mean()/50

                    visax.append(current_barb(vis_x, vis_y,vis_U,vis_V,                                               
                                                color = 'k', ax = ax[0], linewidth = 0.4,
                                                fixed_len = vis_fixed_len,scf = vis_scf))
                    visax.append(current_barb(vis_x+vis_offset, vis_y+vis_offset,vis_U,vis_V,                                               
                                                color = 'w', ax = ax[0], linewidth = 0.4,
                                                fixed_len = vis_fixed_len,scf = vis_scf, linestyle = 'dotted'))
                    """

            ###################################################################################################
            ### Redraw canvas
            ###################################################################################################
            func_but_text_han['waiting'].set_color('w')
            func_but_text_han['waiting'].set_text('Waiting')
            if verbose_debugging: print('Canvas draw', datetime.now())

            if do_timer: timer_lst.append(('Redraw',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            fig.canvas.draw_idle()
            if verbose_debugging: print('Canvas flush', datetime.now())
            fig.canvas.flush_events()
            if verbose_debugging: print('Canvas drawn and flushed', datetime.now())

            # set current axes to hidden full screen axes for click interpretation
            plt.sca(clickax)
            
    
            ###################################################################################################
            ### Runtime stats
            ###################################################################################################

            if do_timer: timer_lst.append(('Redrawn',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            if do_timer: 
                print()
                if do_memory:
                    for i_i in range(1,len(timer_lst)):print('Stage time %02i - %02i: %s (%i dMB, %i MB) - %s - %s '%(i_i-1,i_i,timer_lst[i_i][1] - timer_lst[i_i-1][1],timer_lst[i_i][2] - timer_lst[i_i-1][2],timer_lst[i_i][2], timer_lst[i_i-1][0],timer_lst[i_i][0]))
                else:
                    for i_i in range(1,len(timer_lst)):print('Stage time %02i - %02i: %s - %s - %s '%(i_i-1,i_i,timer_lst[i_i][1] - timer_lst[i_i-1][1], timer_lst[i_i-1][0],timer_lst[i_i][0]))
                print()
                
                print('Stage time 1 - End: %s'%(timer_lst[-1][1] - timer_lst[0][1]))
                if verbose_debugging: print()

                print('Button Press - End: %s'%(timer_lst[-1][1] - timer_lst[1][1]))
                if verbose_debugging: print()


            if do_timer: timer_lst = []

            if do_timer: timer_lst.append(('Timer reset',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            
            #await click with ginput
            if verbose_debugging: print('Waiting for button press', datetime.now())
            if verbose_debugging: print('mode', mode,'mouse_in_Click',mouse_in_Click,datetime.now())
            

            ###################################################################################################
            ### if click mode, ginput
            ###################################################################################################

            
            if secondary_fig is not None:
                #while plt.fignum_exists(figts.number):
                #    time.sleep(1)
                if secondary_fig:
                    time.sleep(5)
                    secondary_fig = False
            #pdb.set_trace()# for ss in locals().keys(): print(ss)
            if mode == 'Loop':
                if mouse_in_Click:
                    mode = 'Click'
                    but_name = 'Click'
                    func_but_text_han['Click'].set_color('gold')
                    func_but_text_han['Loop'].set_color('k')
            if mode == 'Click':
                #if verbose_debugging: print('mode Click, check justplot:',justplot, datetime.now())
                if justplot == False:
                    
                    #if verbose_debugging: print('justplot false, ginput:',justplot, datetime.now())
                    
                    #tmp_press = plt.ginput(1)
                    buttonpress = True
                    while buttonpress: buttonpress = plt.waitforbuttonpress()
                    tmp_press = [[mouse_info['xdata'],mouse_info['ydata']]]
                    del(buttonpress)
                    #pdb.set_trace()
                    #mouse_info = {'button':event.button,'x':event.x, 'y':event.y, 'xdata':event.xdata, 'ydata':event.ydata}
            # if tmp_press is empty (button press detected from another window, persist previous location. 
            #    Previously a empty array led to a continue, which led to the bug where additional colorbar were added
            if len(tmp_press) == 0:
                press_ginput = press_ginput
                button_press = False
            else:
                press_ginput = tmp_press
                button_press = True

            

            if verbose_debugging: print('button_press',button_press)
            if verbose_debugging: print('')
            if verbose_debugging: print('')
            if verbose_debugging: print('')
            if verbose_debugging: print('Button pressed!', datetime.now())
            if do_timer: timer_lst.append(('Button pressed',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            
            ###################################################################################################
            ### Waiting Label
            ###################################################################################################

            if do_timer: timer_lst.append(('Waiting draw_idle',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            func_but_text_han['waiting'].set_color('r')
            fig.canvas.draw()
            if verbose_debugging: print('Canvas flush wait label', datetime.now())
            fig.canvas.flush_events()
            if verbose_debugging: print('Canvas drawn and flushed wait label', datetime.now())

            ###################################################################################################
            ### Find where clicked
            ###################################################################################################

            clii,cljj = press_ginput[0][0],press_ginput[0][1]
                
            
            ###################################################################################################
            ### If justplot, hijack code
            ###################################################################################################

            if do_timer: timer_lst.append(('Just Plot',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            if justplot:
                
                clickax.set_axis_off()
                clickax.remove()
                clickax = fig.add_axes([0,0,0.001,0.001], frameon=False)
                clickax.axis('off')
   

                save_figure_funct()

                if just_plt_cnt == len(just_plt_vals): return 


                clii,cljj  = 0,0
                secdataset_proc = just_plt_vals[just_plt_cnt][0]
                tmp_date_in_ind = just_plt_vals[just_plt_cnt][1]
                z_meth = just_plt_vals[just_plt_cnt][2]
                zz = just_plt_vals[just_plt_cnt][3]
                reload_map = just_plt_vals[just_plt_cnt][4]
                reload_ew = just_plt_vals[just_plt_cnt][5]
                reload_ns = just_plt_vals[just_plt_cnt][6]
                reload_hov = just_plt_vals[just_plt_cnt][7]
                reload_ts = just_plt_vals[just_plt_cnt][8]
                try:
                    tmp_date_in_ind_ind = int(tmp_date_in_ind)
                except:
                    pdb.set_trace()

                if tmp_date_in_ind_ind > 10000:
                    jp_date_in_ind_datetime = datetime.strptime(tmp_date_in_ind,date_fmt)
                    jp_date_in_ind_datetime_timedelta = np.array([(ss - jp_date_in_ind_datetime).total_seconds() for ss in time_datetime])
                    ti = np.abs(jp_date_in_ind_datetime_timedelta).argmin()
                else:
                    ti = int(tmp_date_in_ind)
                if verbose_debugging: print('Setting justplot secdataset_proc: %s'%(secdataset_proc), datetime.now())
                if verbose_debugging: print('Setting justplot ti from date_in_ind (%s): ti = %i (%s). '%(date_in_ind,ti, time_datetime[ti]), datetime.now())
                if verbose_debugging: print('Setting just_plt_vals: ',just_plt_vals[just_plt_cnt], datetime.now())
                
                print('\n\njust_plt_cnt,njust_plt_cnt:\n\n',just_plt_cnt,njust_plt_cnt)
                
                just_plt_cnt += 1


            
            ###################################################################################################
            ### get and set current xylims
            ###################################################################################################



            if do_timer: timer_lst.append(('set current xylims',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            if verbose_debugging: print("selected clii = %f,cljj = %f"%(clii,cljj))

            #get click location, and current axis limits for ax[0], and set them
            # defunct? was trying to allow zooming
            cur_xlim = np.array(ax[0].get_xlim())
            cur_ylim = np.array(ax[0].get_ylim())

            ax[0].set_xlim(cur_xlim)
            ax[0].set_ylim(cur_ylim)



            
            
            ###################################################################################################
            ### Get click coords
            ###################################################################################################
            if do_timer: timer_lst.append(('Get click coords',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

            #find clicked axes:
            is_in_axes = False
            
            # convert the mouse click into data indices, and report which axes was clicked
            '''
            try:
                sel_ax,sel_ii,sel_jj,sel_ti,sel_zz, sel_xlocval,sel_ylocval = indices_from_ginput_ax(ax,clii,cljj, thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
            except:
                print('indices_from_ginput_ax failed',clii,cljj )
                pdb.set_trace()
            '''
            sel_ax,sel_ii,sel_jj,sel_ti,sel_zz, sel_xlocval,sel_ylocval = indices_from_ginput_ax(ax,clii,cljj, thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
            
                
            if verbose_debugging: print("selected sel_ax = %s,sel_ii = %s,sel_jj = %s,sel_ti = %s,sel_zz = %s"%(sel_ax,sel_ii,sel_jj,sel_ti,sel_zz))

            #print(sel_ax,sel_ii,sel_jj,sel_ti,sel_zz )

            if sel_ax is not None :  is_in_axes = True 

            
            ###################################################################################################
            ### If axes clicked, change ind, decide what data to reload
            ###################################################################################################
            if do_timer: timer_lst.append(('If axes, change ind',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            if verbose_debugging: print('Interpret Mouse click: figure axes, location change', datetime.now())

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
                
                if z_meth_default == 'z_index':
                    zi = np.abs(grid_dict['Dataset 1']['gdept'][:,jj,ii] - sel_zz).argmin()

                
                reload_map = True
                reload_ts = True

            elif sel_ax in [4]:
                # if in hov/time series, change map, and slices
                ti = sel_ti
                
                reload_map = True
                reload_ew = True
                reload_ns = True

                if do_Obs:
                    reload_Obs = True
                if do_MLD:
                    reload_MLD = True
   
            
            if mode == 'Loop':
                ti+=1
                if ti == ntime: 
                    ti = 0
                    #mode = 'Click'
                    #pdb.set_trace()
                    

                    
            
            if verbose_debugging: print('Decide what to reload', datetime.now())

            if verbose_debugging: print("selected ii = %s,jj = %s,ti = %s,zz = %s"%(ii,jj,ti,zz))

            # if in button, change variables. 
            if verbose_debugging: print('Interpret Mouse click: Change Variable', datetime.now())

            
            
            ###################################################################################################
            ### If var clicked, change var
            ###################################################################################################

            if do_timer: timer_lst.append(('If not in axes',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
           
            if not is_in_axes: 
                if do_timer: timer_lst.append(('Check Var',datetime.now()))
                if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
           
                for but_name in but_extent.keys():
                    
                    but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = but_extent[but_name]
                    if (clii >= but_pos_x0) & (clii <= but_pos_x1) & (cljj >= but_pos_y0) & (cljj <= but_pos_y1):
                        is_in_axes = True
                        if but_name in var_but_mat:
                            var = but_name

                            func_but_text_han['waiting'].set_text('Waiting:\n' + but_name)

                            # redraw canvas
                            fig.canvas.draw_idle()
                            
                            #flush canvas
                            fig.canvas.flush_events()

                            if var_dim[var] == 3:
                                z_meth = z_meth_default

                                func_but_text_han['Depth level'].set_color('r')
                                func_but_text_han['Surface'].set_color('k')
                                func_but_text_han['Near-Bed'].set_color('k')
                                func_but_text_han['Surface-Bed'].set_color('k')
                                func_but_text_han['Depth-Mean'].set_color('k')
                            
                            for vi,var_dat in enumerate(var_but_mat): but_text_han[var_dat].set_color('k')
                            but_text_han[but_name].set_color('r')

                            # redraw canvas
                            fig.canvas.draw_idle()
                            
                            #flush canvas
                            fig.canvas.flush_events()
                            
                            climnorm = None 

                            reload_map = True
                            reload_ew = True
                            reload_ns = True
                            reload_hov = True
                            reload_ts = True
                            if do_Obs:
                                reload_Obs = True

                ###################################################################################################
                ### If function clicked, call function
                ###################################################################################################
                if do_timer: timer_lst.append(('Check func',datetime.now()))
                if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            
                if verbose_debugging: print('Interpret Mouse click: Functions', datetime.now())
                for but_name in func_but_extent.keys():
                    
                    but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = func_but_extent[but_name]
                    if (clii >= but_pos_x0) & (clii <= but_pos_x1) & (cljj >= but_pos_y0) & (cljj <= but_pos_y1):
                        is_in_axes = True
                        print('but_name:',but_name)

                        func_but_text_han['waiting'].set_text('Waiting:\n' + but_name)

                        # redraw canvas
                        fig.canvas.draw_idle()
                        
                        #flush canvas
                        fig.canvas.flush_events()


                        if but_name in 'Reset zoom':
                            # set xlim and ylim to max size possible from lat_d[1] and nav_lon
                            cur_xlim = np.array([lon_d[1].min(),lon_d[1].max()])
                            cur_ylim = np.array([lat_d[1].min(),lat_d[1].max()])
                            zlim_max = None
                        elif but_name in 'Zoom':
                            # use ginput to take two clicks as zoom region. 
                            # only coded for main axes
                            if mouse_info['button'].name == 'MIDDLE':
                                cur_xlim = np.array([lon_d[1].min(),lon_d[1].max()])
                                cur_ylim = np.array([lat_d[1].min(),lat_d[1].max()])
                                zlim_max = None
                            else:

                                tmp_zoom_in = True
                                if mouse_info['button'].name == 'RIGHT':tmp_zoom_in = False
                                
                                plt.sca(clickax)
                                #tmpzoom0 = plt.ginput(1)

                                buttonpress = True
                                while buttonpress: buttonpress = plt.waitforbuttonpress()
                                tmpzoom0 = [[mouse_info['xdata'],mouse_info['ydata']]]
                                del(buttonpress)


                                if tmp_zoom_in: func_but_text_han['Zoom'].set_color('r')
                                elif not tmp_zoom_in: func_but_text_han['Zoom'].set_color('g')
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()
                                zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz,zoom0_sel_xlocval,zoom0_sel_ylocval = indices_from_ginput_ax(ax,tmpzoom0[0][0],tmpzoom0[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                if zoom0_ax in [1,2,3]:
                                    zlim_max = zoom0_zz
                                elif zoom0_ax in [0]:
                                    #tmpzoom1 = plt.ginput(1)
                                    buttonpress = True
                                    while buttonpress: buttonpress = plt.waitforbuttonpress()
                                    tmpzoom1 = [[mouse_info['xdata'],mouse_info['ydata']]]
                                    del(buttonpress)

                                    zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz, zoom1_sel_xlocval,zoom1_sel_ylocval = indices_from_ginput_ax(ax,tmpzoom1[0][0],tmpzoom1[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                        
                                    if verbose_debugging: print(zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz)
                                    if verbose_debugging: print(zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz)
                                    if verbose_debugging: print(cur_xlim)
                                    if verbose_debugging: print(cur_ylim)

                                    # if both clicks in main axes, use clicks for the new x and ylims
                                    if (zoom0_ax is not None) & (zoom1_ax is not None):
                                        if zoom0_ax == zoom1_ax:
                                            if zoom0_ax == 0:
                                                #cl_cur_xlim = np.array([lon_d[1][zoom0_jj,zoom0_ii],lon_d[1][zoom1_jj,zoom1_ii]])
                                                #cl_cur_ylim = np.array([lat_d[1][zoom0_jj,zoom0_ii],lat_d[1][zoom1_jj,zoom1_ii]])

                                                cl_cur_xlim = np.array([zoom0_sel_xlocval,zoom1_sel_xlocval])
                                                cl_cur_ylim = np.array([zoom0_sel_ylocval,zoom1_sel_ylocval])
                                                cl_cur_xlim.sort()
                                                cl_cur_ylim.sort()
                                                if tmp_zoom_in:
                                                    cur_xlim = cl_cur_xlim
                                                    cur_ylim = cl_cur_ylim
                                                else:
                                                    # If right click (initially, or last time), zoom out. 
                                                    #pdb.set_trace()
                                                    #current width of the x and y axis
                                                    #dcur_xlim = cur_xlim.ptp()
                                                    #dcur_ylim = cur_ylim.ptp()
                                                    dcur_xlim = np.ptp(cur_xlim)
                                                    dcur_ylim = np.ptp(cur_ylim)
                                                    #middle of current the x and y axis
                                                    mncur_xlim = cur_xlim.mean()
                                                    mncur_ylim = cur_ylim.mean()
                                                    #width of the clicked x and y points
                                                    #dcur_cl_xlim = cl_cur_xlim.ptp()
                                                    #dcur_cl_ylim = cl_cur_ylim.ptp()
                                                    dcur_cl_xlim = np.ptp(cl_cur_xlim)
                                                    dcur_cl_ylim = np.ptp(cl_cur_ylim)
                                                    
                                                    # scale up axis width with dcur_xlim/dcur_cl_xlim, and centre.
                                                    cur_xlim = dcur_xlim*(dcur_xlim/dcur_cl_xlim)*np.array([-0.5,0.5])+mncur_xlim
                                                    cur_ylim = dcur_ylim*(dcur_ylim/dcur_cl_ylim)*np.array([-0.5,0.5])+mncur_ylim


                                
                                if verbose_debugging: print(cur_xlim)
                                if verbose_debugging: print(cur_ylim)
                                func_but_text_han['Zoom'].set_color('k')
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()
                                                
                                                
                        elif but_name == 'Axis':
                            if axis_scale == 'Auto':

                                func_but_text_han['Axis'].set_text('Axis: Equal')
                                ax[0].axis('equal')
                                axis_scale = 'Equal'
                            elif axis_scale == 'Equal':

                                func_but_text_han['Axis'].set_text('Axis: Auto')
                                axis_scale = 'Auto'
                                ax[0].axis('auto')
                                #cur_xlim = np.array([lon_d[1].min(),lon_d[1].max()])
                                #cur_ylim = np.array([lat_d[1].min(),lat_d[1].max()])
                            '''
                        elif but_name == 'Clim: Reset':
                            clim = None

                            '''
                        elif but_name == 'MLD':
                            if mouse_info['button'].name == 'LEFT':
                                if MLD_show == True:

                                    func_but_text_han['MLD'].set_color('0.5')
                                    MLD_show = False
                                elif MLD_show == False:

                                    func_but_text_han['MLD'].set_color('k')
                                    MLD_show = True




                            elif mouse_info['button'].name == 'RIGHT':

                                # Bring up a options window for Obs
                                
                                # button names                            
                                mld_but_names = MLD_var_lst + ['Close']
                                
                                
                                # button switches  
                                mld_but_sw = {}
                                #obs_but_sw['Hide_Obs'] = {'v':Obs_hide, 'T':'Show Obs','F': 'Hide Obs'}
                                #obs_but_sw['Edges'] = {'v':Obs_hide, 'T':'Show Edges','F': 'Hide Edges'}
                                #obs_but_sw['Loc'] = {'v':Obs_hide, 'T':"Don't Selected point",'F': 'Move Selected point'}
                                #for m_var in MLD_var_lst:  mld_but_sw[m_var] = {'v':m_var == MLD_var ,'T': m_var + ' selected','F':'choose ' +m_var,'T_col': 'k','F_col':'0.5'}
                                for m_var in MLD_var_lst:  mld_but_sw[m_var] = {'v':m_var == MLD_var ,'T': m_var ,'F': m_var,'T_col': 'r','F_col':'k'}

                                mldbut_sel = pop_up_opt_window(mld_but_names, opt_but_sw = mld_but_sw)

                                

                                # Set the main figure and axis to be current
                                plt.figure(fig.figure)
                                plt.sca(clickax)

                                if mldbut_sel in MLD_var_lst:
                                    MLD_var = mldbut_sel
                                    reload_MLD = True
                                    reload_ew = True
                                    reload_ns = True
                                    print('mldbut_sel:',mldbut_sel)
                                    print('MLD_var:',MLD_var)
                                    print('reload_MLD:',reload_MLD)

                                # if the button closed was one of the Obs types, add or remove from the hide list
                                #for m_var in MLD_var_lst:  
                                #    if mld_but_sw[m_var]['v']:
                                #        MLD_var = m_var
                                        
                        

                        elif but_name == 'Help':
                        
                            print('      mouse info:  '  )
                            print(mouse_info)
                            print(mouse_info['button'])
                            print(mouse_info['button'].name)
                            print('--------------------')
                            if mouse_info['button'].name == 'LEFT':


                                # select the Help with ginput
                                tmphelploc = plt.ginput(1)

                                # convert to the nearest model grid box                            
                                helploc_ii, helploc_jj, = tmphelploc[0][0],tmphelploc[0][1]
                                help_ax,help_ii,help_jj,help_ti,help_zz, sel_xlocval,sel_ylocval = indices_from_ginput_ax(ax,helploc_ii, helploc_jj, thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                #
                                help_type = 'None'
                                help_but = ''
                                    
                                if help_ax is None:

                                    if verbose_debugging: print('Interpret Mouse click: Functions', datetime.now())

                                    ###################################################################################################
                                    ### See if func clicked
                                    ###################################################################################################

                                    for but_name in func_but_extent.keys():
                                        
                                        but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = func_but_extent[but_name]
                                        if (helploc_ii >= but_pos_x0) & (helploc_ii <= but_pos_x1) & (helploc_jj >= but_pos_y0) & (helploc_jj <= but_pos_y1):
                                            help_type = 'func'
                                            help_but = but_name
                                    
                                    ###################################################################################################
                                    ### See var func clicked
                                    ###################################################################################################
                                    
                                    for but_name in but_extent.keys():
                                        
                                        but_pos_x0,but_pos_x1,but_pos_y0,but_pos_y1 = but_extent[but_name]
                                        if (helploc_ii >= but_pos_x0) & (helploc_ii <= but_pos_x1) & (helploc_jj >= but_pos_y0) & (helploc_jj <= but_pos_y1):
                                            if but_name in var_but_mat:
                                                help_type = 'var'
                                                help_but = but_name
                                    
                                else:

                                    help_type = 'axis'
                                    help_but = 'axis: %s'% letter_mat[help_ax]

                                print(help_type,help_but)

                                help_text = get_help_text(help_type,help_but)
                                pop_up_info_window(help_text)

                                # Set the main figure and axis to be current
                                plt.figure(fig.figure)
                                plt.sca(clickax)
                                
                        elif but_name == 'Obs':
                            #print('      mouse info:  '  )
                            #print(mouse_info)
                            #print(mouse_info['button'])
                            #print(mouse_info['button'].name)
                            #print('--------------------')

                            if mouse_info['button'].name == 'LEFT':

                                # predefine dictionaries

                                (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst)
                                # select the observation with ginput
                                tmpobsloc = plt.ginput(1)

                                # convert to the nearest model grid box
                                
                                obs_ax,obs_ii,obs_jj,obs_ti,obs_zz, sel_xlocval,sel_ylocval = indices_from_ginput_ax(ax,tmpobsloc[0][0],tmpobsloc[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                            
                                
                                # if the main map axis is selected, continue,
                                if obs_ax == 0:
                                    #if True:
                                    
                                    # extract lat and lon,

                                    obs_lon = sel_xlocval # lon_d[1][obs_jj,obs_ii]
                                    obs_lat = sel_ylocval # lat_d[1][obs_jj,obs_ii]

                                    if Obs_pair_loc:
                                        ii = obs_ii
                                        jj = obs_jj

                                        # and reload slices, and hovmuller/time series
                                        reload_ew = True
                                        reload_ns = True
                                        reload_hov = True
                                        reload_ts = True

                                    # set some tmp dictionaries
                                    #(obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                    #    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst, Fill = False)
                                    
                                    tmpobs_dist_sel = {}
                                    (tmpobs_z_sel,tmpobs_obs_sel,tmpobs_mod_sel,tmpobs_lon_sel,tmpobs_lat_sel,
                                        tmpobs_stat_id_sel,tmpobs_stat_type_sel,tmpobs_stat_time_sel) = obs_reset_sel(Dataset_lst, Fill = False)

                                    
                                    # cycle through available Obs types
                                    for ob_var in Obs_var_lst_sub:
                                        tmpobs_z_sel[ob_var] = {}
                                        tmpobs_obs_sel[ob_var] = {}
                                        tmpobs_mod_sel[ob_var] = {}
                                        tmpobs_lon_sel[ob_var] = {}
                                        tmpobs_lat_sel[ob_var] = {}
                                        tmpobs_dist_sel[ob_var] = {}

                                        tmpobs_stat_id_sel[ob_var] = {}
                                        tmpobs_stat_type_sel[ob_var] = {}
                                        tmpobs_stat_time_sel[ob_var] = {}
                                        # and data sets.
                                        for tmp_datstr in Dataset_lst:
                                            # extract Obs: lon, lat, z OBS, mod,, stations info 
                                            tmpobsx = Obs_dat_dict[tmp_datstr][ob_var]['LONGITUDE']
                                            tmpobsy = Obs_dat_dict[tmp_datstr][ob_var]['LATITUDE']
                                            tmpobs_obs = Obs_dat_dict[tmp_datstr][ob_var]['OBS'][:] 
                                            tmpobs_mod = Obs_dat_dict[tmp_datstr][ob_var]['MOD_HX'][:] 
                                            tmpobs_z = Obs_dat_dict[tmp_datstr][ob_var]['DEPTH'][:] 
                                            tmpobs_stat_id = Obs_dat_dict[tmp_datstr][ob_var]['STATION_IDENTIFIER'][:] 
                                            tmpobs_stat_type = Obs_dat_dict[tmp_datstr][ob_var]['STATION_TYPE'][:] 
                                            tmpobs_stat_time = Obs_dat_dict[tmp_datstr][ob_var]['JULD_datetime'][:] 
                                            tmpobs_z = Obs_dat_dict[tmp_datstr][ob_var]['DEPTH'][:] 

                                            # if Obs data is 2d data, reshape to match 3d profile data. 
                                            if len(tmpobs_obs.shape) == 1:

                                                tmpobs_obs = tmpobs_obs.reshape(-1,1)
                                                tmpobs_mod = tmpobs_mod.reshape(-1,1)
                                                tmpobs_z = tmpobs_z.reshape(-1,1)
                                            # find distance from selected point to all obs in this Obs type    
                                            tmp_obs_dist = np.sqrt((tmpobsx - obs_lon)**2 +(tmpobsy - obs_lat)**2)
                                        
                                            # find the minimum value and index
                                            tmp_obs_obs_dist = tmp_obs_dist.min()
                                            tmp_obs_obs_ind = tmp_obs_dist.argmin()

                                            #record the profile and distance for that Obs
                                            tmpobs_z_sel[ob_var][tmp_datstr] = tmpobs_z[tmp_obs_obs_ind]
                                            tmpobs_obs_sel[ob_var][tmp_datstr] = tmpobs_obs[tmp_obs_obs_ind]
                                            tmpobs_mod_sel[ob_var][tmp_datstr] = tmpobs_mod[tmp_obs_obs_ind]
                                            tmpobs_lon_sel[ob_var][tmp_datstr] = tmpobsx[tmp_obs_obs_ind]
                                            tmpobs_lat_sel[ob_var][tmp_datstr] = tmpobsy[tmp_obs_obs_ind]
                                            #pdb.set_trace()
                                            tmpobs_stat_id_sel[ob_var][tmp_datstr] = tmpobs_stat_id[tmp_obs_obs_ind].strip()
                                            tmpobs_stat_type_sel[ob_var][tmp_datstr] = tmpobs_stat_type[tmp_obs_obs_ind]
                                            tmpobs_stat_time_sel[ob_var][tmp_datstr] = tmpobs_stat_time[tmp_obs_obs_ind]

                                            tmpobs_dist_sel[ob_var] = tmp_obs_obs_dist
                                    
                                    # put all distances into one array
                                    obs_dist_sel_cf_mat = np.array([tmpobs_dist_sel[ob_var] for ob_var in Obs_var_lst_sub])

                                    if (obs_dist_sel_cf_mat).size >0:
                                        
                                        # select the Obs Obs type closest to the selected point
                                        sel_Obs_var = Obs_var_lst_sub[obs_dist_sel_cf_mat.argmin()]
                                        
                                        #select obs data from Obs type closest to the selected point
                                        obs_z_sel = tmpobs_z_sel[sel_Obs_var]
                                        obs_obs_sel = tmpobs_obs_sel[sel_Obs_var]
                                        obs_mod_sel = tmpobs_mod_sel[sel_Obs_var]
                                        obs_lon_sel = tmpobs_lon_sel[sel_Obs_var]
                                        obs_lat_sel = tmpobs_lat_sel[sel_Obs_var]

                                        obs_stat_id_sel = tmpobs_stat_id_sel[sel_Obs_var]
                                        obs_stat_type_sel = tmpobs_stat_type_sel[sel_Obs_var]
                                        obs_stat_time_sel = tmpobs_stat_time_sel[sel_Obs_var]


                                        del(tmpobs_z_sel)
                                        del(tmpobs_obs_sel)
                                        del(tmpobs_mod_sel)
                                        del(tmpobs_lon_sel)
                                        del(tmpobs_lat_sel)
                                        del(tmpobs_dist_sel)

                                        del(tmpobs_stat_id_sel)
                                        del(tmpobs_stat_type_sel)
                                        del(tmpobs_stat_time_sel)


                                        # append this data to an array to help select x and y lims
                                        for tmp_datstr in Dataset_lst:
                                            tmp_pf_ylim_dat = np.ma.append(np.array(tmp_py_ylim),obs_z_sel[tmp_datstr])
                                            #tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel)[tmp_datstr],obs_obs_sel)
                                            tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel[tmp_datstr]),obs_obs_sel[tmp_datstr])
                                        
                                    else:
                                        
                                        (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                            obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst)

                            elif mouse_info['button'].name == 'RIGHT':

                                # Bring up a options window for Obs
                                
                                # button names                            
                                obs_but_names = ['ProfT','SST_ins','SST_sat','ProfS','SLA','ChlA','Hide_Obs','Edges','Loc','AbsAnom','Close']
                                
                                
                                # button switches  
                                obs_but_sw = {}
                                obs_but_sw['Hide_Obs'] = {'v':Obs_hide, 'T':'Show Obs','F': 'Hide Obs'}
                                #obs_but_sw['Edges'] = {'v':Obs_hide, 'T':'Show Edges','F': 'Hide Edges'}
                                #obs_but_sw['Loc'] = {'v':Obs_hide, 'T':"Don't Selected point",'F': 'Move Selected point'}
                                obs_but_sw['Edges'] = {'v':Obs_hide_edges, 'T':'Show Edges','F': 'Hide Edges'}
                                obs_but_sw['Loc'] = {'v':Obs_pair_loc, 'T':"Don't Selected point",'F': 'Move Selected point'}
                                obs_but_sw['AbsAnom'] = {'v':Obs_AbsAnom, 'T':"Observed Values",'F': 'Model - Obs'}
                                #for ob_var in Obs_var_lst_sub:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var + ' hidden'}
                                #for ob_var in Obs_varlst:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var + ' hidden'}
                                for ob_var in Obs_var_lst_sub:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var,'T_col':'k','F_col':'0.5'}
                                #for ob_var in Obs_var_lst_sub:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var + ' hidden'}
                                for ob_var in Obs_varlst:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var,'T_col':'k','F_col':'0.5'}

                                obbut_sel = pop_up_opt_window(obs_but_names, opt_but_sw = obs_but_sw)


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
                                if obbut_sel == 'AbsAnom':      Obs_AbsAnom = not Obs_AbsAnom



                                reload_Obs = True
                        

                        elif but_name == 'Xsect':
                            
                            xsect_secdataset_proc = secdataset_proc
                            tmpxsect_secdataset_proc_oper = xsect_secdataset_proc[4]
                            if tmpxsect_secdataset_proc_oper == '-': 
                                xsect_secdataset_proc = Dataset_lst[0]

                            if (mouse_info['button'].name == 'RIGHT') | (loaded_xsect == False):
                                loaded_xsect = True

                                tmp_xsect_jjii_npnt = 0
                                while tmp_xsect_jjii_npnt<2:
                                    xsectloc_lst = plt.ginput(-1)
                                    tmp_xsect_jjii_npnt = len(xsectloc_lst)
                                    if tmp_xsect_jjii_npnt<2:
                                        print('Xsect: you must select at least 2 points. You have selected %i points'%tmp_xsect_jjii_npnt)
                                        print('Xsect: start selection again')
                                #pdb.set_trace()
                                del(tmp_xsect_jjii_npnt)

                                print('Xsect: ginput exited')
                                xs0 = datetime.now()

                                # convert to the nearest model grid box
                                xsect_ax_pnt_lst = []
                                xsect_ii_pnt_lst = []
                                xsect_jj_pnt_lst = []
                                
                                for tmpxsectloc in xsectloc_lst: 
                                    tmpxsect_ax,tmpxsect_ii,tmpxsect_jj,tmpxsect_ti,tmpxsect_zz, tmpxsect_sel_xlocval,tmpxsect_sel_ylocval = indices_from_ginput_ax(ax,tmpxsectloc[0],tmpxsectloc[1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                    if (tmpxsect_ax is None)|(tmpxsect_ii is None)|(tmpxsect_jj is None): 
                                        print('Xsect: selected point outside axis, skipping')
                                        continue
                                    xsect_ax_pnt_lst.append(tmpxsect_ax)
                                    xsect_ii_pnt_lst.append(tmpxsect_ii)
                                    xsect_jj_pnt_lst.append(tmpxsect_jj)

                                
                                xsect_jjii_npnt = len(xsect_ii_pnt_lst)
                                if xsect_jjii_npnt < 2: 
                                    print('Xsect: you must select at least 2 points. You have selected %i points'%xsect_jjii_npnt)
                                    print('Xsect: exiting Xsect')
                                    continue
                                    

                                
                                sec_th_d_ind = int(xsect_secdataset_proc[8:]) # int(tmp_datstr[-1])
                                #pdb.set_trace()

                                #convert selected indices into Lat and Lon 
                                #   indices_from_ginput_ax hard coded to return indices for Dataset 1.

                                xsect_lon_pnt_mat = lon_d[1][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]
                                xsect_lat_pnt_mat = lat_d[1][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]
                                '''
                                try:
                                    xsect_lon_pnt_mat = lon_d[sec_th_d_ind][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]
                                    xsect_lat_pnt_mat = lat_d[sec_th_d_ind][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]
                                except:
                                    pdb.set_trace()
                                '''
                                xsect_lon_dict = {}
                                xsect_lat_dict = {}
                                xsect_lon_pnt_dict = {}
                                xsect_lat_pnt_dict = {}
                                xsect_ii_pnt_dict = {}
                                xsect_jj_pnt_dict = {}
                                xsect_jj_ind_dict = {}
                                xsect_ii_ind_dict = {}
                                xsect_pnt_ind_dict = {}
                                nxsect_dict = {}

                                for tmp_datstr in Dataset_lst:

                                    th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                                    
                                    #xsect_lon_pnt_mat = lon_d[th_d_ind][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]
                                    #xsect_lat_pnt_mat = lat_d[th_d_ind][[xsect_jj_pnt_lst],[xsect_ii_pnt_lst]][0,:]



                                    tmpnlat, tmpnlon = domsize[th_d_ind] # lat_d[th_d_ind].shape
                                    #th_d_ind1 = int(tmp_datstr1[-1])
                                    #t#h_d_ind1 = int(tmp_datstr1[8:])

                                    xs1 = datetime.now()
                                    #if configd[int(secdataset_proc[8:])].lower() in ['amm7','amm15','co9p2','gulf18','orca025']:
                                    #if True:

                                    xsect_ii_ind_lst = []
                                    xsect_jj_ind_lst = []
                                    xsect_n_ind_lst = []
                                    tmp_xsect_ii_pnt_dict = []
                                    tmp_xsect_jj_pnt_dict = []

                                    #convert selected  points into lon lat points for current grid
                                    for xi in range(xsect_jjii_npnt):

                                        #pdb.set_trace()
                                        tmp_jj_jjii_from_lon_lat,tmp_ii_jjii_from_lon_lat = jjii_from_lon_lat(xsect_lon_pnt_mat[xi],xsect_lat_pnt_mat[xi],lon_d[th_d_ind],lat_d[th_d_ind])
                                        #tmp_lonlatijdist_iijj = np.sqrt((xsect_lon_pnt_mat[xi] - lon_d[th_d_ind])**2 + (xsect_lat_pnt_mat[xi] - lat_d[th_d_ind])**2)).argmin()
                                        tmp_xsect_ii_pnt_dict.append(tmp_ii_jjii_from_lon_lat)
                                        tmp_xsect_jj_pnt_dict.append(tmp_jj_jjii_from_lon_lat)
                                    xsect_ii_pnt_dict[tmp_datstr] = np.array(tmp_xsect_ii_pnt_dict)
                                    xsect_jj_pnt_dict[tmp_datstr] = np.array(tmp_xsect_jj_pnt_dict)
                                    del(tmp_xsect_ii_pnt_dict)
                                    del(tmp_xsect_jj_pnt_dict)

                                    #pdb.set_trace()
                                    for xi in range(xsect_jjii_npnt-1):

                                        #pdb.set_trace()

                                        #tmp_ii = (xsect_lon_pnt_mat[xi] - lon_d[th_d_ind]**2) + (xsect_lat_pnt_mat[xi] - lat_d[th_d_ind]**2)
                                        #tmp_xsect_ii_ind,tmp_xsect_jj_ind = profile_line( xsect_ii_pnt_lst[xi:xi+2],xsect_jj_pnt_lst[xi:xi+2], ni = tmpnlat )
                                        tmp_xsect_ii_ind,tmp_xsect_jj_ind = profile_line( xsect_ii_pnt_dict[tmp_datstr][xi:xi+2],xsect_jj_pnt_dict[tmp_datstr][xi:xi+2], ni = tmpnlat )
                                        xsect_ii_ind_lst.append(tmp_xsect_ii_ind)
                                        xsect_jj_ind_lst.append(tmp_xsect_jj_ind)
                                        xsect_n_ind_lst.append(tmp_xsect_ii_ind.size)

                                    xsect_ii_ind_mat = np.concatenate(xsect_ii_ind_lst)
                                    xsect_jj_ind_mat = np.concatenate(xsect_jj_ind_lst)
                                    nxsect = xsect_ii_ind_mat.size

                                    xsect_lon_mat = lon_d[th_d_ind][[xsect_jj_ind_mat],[xsect_ii_ind_mat]][0,:]
                                    xsect_lat_mat = lat_d[th_d_ind][[xsect_jj_ind_mat],[xsect_ii_ind_mat]][0,:]
                                    
                                    xsect_pnt_ind = np.append(0,np.cumsum(xsect_n_ind_lst)-1)


                                    xsect_lon_dict[tmp_datstr] = xsect_lon_mat
                                    xsect_lat_dict[tmp_datstr] = xsect_lat_mat
                                    xsect_lon_pnt_dict[tmp_datstr] = xsect_lon_pnt_mat
                                    xsect_lat_pnt_dict[tmp_datstr] = xsect_lat_pnt_mat
                                    xsect_ii_ind_dict[tmp_datstr] = xsect_ii_ind_mat
                                    xsect_jj_ind_dict[tmp_datstr] = xsect_jj_ind_mat
                                    xsect_pnt_ind_dict[tmp_datstr] = xsect_pnt_ind
                                    nxsect_dict[tmp_datstr] = nxsect

                                    
                                """

                                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

                                xs1 = datetime.now()
                                #if configd[int(secdataset_proc[8:])].lower() in ['amm7','amm15','co9p2','gulf18','orca025']:
                                #if True:

                                xsect_ii_ind_lst = []
                                xsect_jj_ind_lst = []
                                xsect_n_ind_lst = []
                                #pdb.set_trace()
                                for xi in range(xsect_jjii_npnt-1):
                                    tmp_xsect_ii_ind,tmp_xsect_jj_ind = profile_line( xsect_ii_pnt_lst[xi:xi+2],xsect_jj_pnt_lst[xi:xi+2], ni = tmpnlat )
                                    xsect_ii_ind_lst.append(tmp_xsect_ii_ind)
                                    xsect_jj_ind_lst.append(tmp_xsect_jj_ind)
                                    xsect_n_ind_lst.append(tmp_xsect_ii_ind.size)

                                xsect_ii_ind_mat = np.concatenate(xsect_ii_ind_lst)
                                xsect_jj_ind_mat = np.concatenate(xsect_jj_ind_lst)
                                nxsect = xsect_ii_ind_mat.size

                                xsect_lon_mat = lon_d[th_d_ind][[xsect_jj_ind_mat],[xsect_ii_ind_mat]][0,:]
                                xsect_lat_mat = lat_d[th_d_ind][[xsect_jj_ind_mat],[xsect_ii_ind_mat]][0,:]
                                
                                xsect_pnt_ind = np.append(0,np.cumsum(xsect_n_ind_lst)-1)
                                """
                                
                            print('Xsect: indices processed.')


                            #xs_map_ax_lst = [ax[0].plot(xsect_lon_mat,xsect_lat_mat,'r.-')]
                            #xs_map_ax_lst.append(ax[0].plot(xsect_lon_pnt_mat,xsect_lat_pnt_mat,'ko-'))
                            xs_map_ax_lst = [ax[0].plot(xsect_lon_dict[xsect_secdataset_proc],xsect_lat_dict[xsect_secdataset_proc],'r.-')]
                            xs_map_ax_lst.append(ax[0].plot(xsect_lon_pnt_dict[xsect_secdataset_proc],xsect_lat_pnt_dict[xsect_secdataset_proc],'ko-'))

                            fig.canvas.draw_idle()
                            if verbose_debugging: print('Canvas flush', datetime.now())
                            fig.canvas.flush_events()
                            if verbose_debugging: print('Canvas drawn and flushed', datetime.now())

            
                            tmp_xsect_x,tmp_xsect_z,tmp_xsect_dat = {},{},{}
                            tmp_xsect_mld_dat = {}

                            '''
                            for tmp_datstr in Dataset_lst:
                                if var_dim[var] == 4:
                                    tmp_xsect_dat[tmp_datstr] = data_inst[tmp_datstr][:,[xsect_jj_ind_mat],[xsect_ii_ind_mat]][:,0,:]
                                    tmp_xsect_z[tmp_datstr] = grid_dict[tmp_datstr]['gdept'][:,[xsect_jj_ind_mat],[xsect_ii_ind_mat]][:,0,:]
                                    tmp_xsect_x[tmp_datstr] = np.arange(nxsect)
                                elif var_dim[var] == 3:
                                    tmp_xsect_dat[tmp_datstr] = data_inst[tmp_datstr][[xsect_jj_ind_mat],[xsect_ii_ind_mat]][0,:]
                                    tmp_xsect_x[tmp_datstr] = np.arange(nxsect)
                                    #pdb.set_trace()
                            '''
                            #pdb.set_trace()
                            for tmp_datstr in Dataset_lst:
                                if var_dim[var] == 4:
                                    tmp_xsect_dat[tmp_datstr] = data_inst[tmp_datstr][:,[xsect_jj_ind_dict[tmp_datstr]],[xsect_ii_ind_dict[tmp_datstr]]][:,0,:]
                                    tmp_xsect_z[tmp_datstr] = np.array(grid_dict[tmp_datstr]['gdept'])[:,[xsect_jj_ind_dict[tmp_datstr]],[xsect_ii_ind_dict[tmp_datstr]]][:,0,:]
                                    tmp_xsect_x[tmp_datstr] = np.arange(nxsect_dict[tmp_datstr])
                                    if do_MLD:
                                        tmp_xsect_mld_dat[tmp_datstr] = data_inst_mld[tmp_datstr][[xsect_jj_ind_dict[tmp_datstr]],[xsect_ii_ind_dict[tmp_datstr]]][0,:]
                                    
                                elif var_dim[var] == 3:
                                    tmp_xsect_dat[tmp_datstr] = data_inst[tmp_datstr][[xsect_jj_ind_dict[tmp_datstr]],[xsect_ii_ind_dict[tmp_datstr]]][0,:]
                                    tmp_xsect_x[tmp_datstr] = np.arange(nxsect_dict[tmp_datstr])
                                    #pdb.set_trace()



                            if figxs is not None:
                                if plt.fignum_exists(figxs.number):
                                    plt.close(figxs)

                            # if 2 datasets, and 3d var do the if... otherwise if 1 dataset or if 2d dataset, do the else
                            if (nDataset == 2)&(var_dim[var] == 4):
                                if (configd[1] == configd[2]):
                                
                                    figxs = plt.figure()
                                    figxs.set_figheight(10*1.2)
                                    figxs.set_figwidth(8*1.5)
                                    figxs.suptitle('Cross-section: %s'%nice_varname_dict[var], fontsize = 20)
                                    plt.subplots_adjust(top=0.92,bottom=0.05,left=0.05,right=1,hspace=0.25,wspace=0.6)
                                    axxs = [plt.subplot(311),plt.subplot(312),plt.subplot(313)]
                                    paxxs = []
                                    for xi,tmp_datstr in enumerate(Dataset_lst): paxxs.append(axxs[xi].pcolormesh(tmp_xsect_x[tmp_datstr],tmp_xsect_z[tmp_datstr],tmp_xsect_dat[tmp_datstr]))
                                    paxxs.append(axxs[2].pcolormesh(tmp_xsect_x[xsect_secdataset_proc],tmp_xsect_z[xsect_secdataset_proc],tmp_xsect_dat[Dataset_lst[1]] - tmp_xsect_dat[Dataset_lst[0]], cmap = matplotlib.cm.seismic))
                                    for tmpax  in axxs: tmpax.invert_yaxis()
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_title(tmp_datstr)
                                    axxs[2].set_title('%s - %s'%(Dataset_lst[1],Dataset_lst[0]))
                                    xs_ylim = np.array(axxs[0].get_ylim())
                                    xs_xlim = np.array(axxs[0].get_xlim())
                                    xs_ylim[0] = tmp_xsect_z[xsect_secdataset_proc][~(tmp_xsect_x[xsect_secdataset_proc]*tmp_xsect_z[xsect_secdataset_proc]*tmp_xsect_dat[xsect_secdataset_proc]).mask].max()
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_ylim(xs_ylim)
                                    axxs[2].set_ylim(xs_ylim)
                                    
                                    if do_MLD:
                                        for xi,tmp_datstr in enumerate(Dataset_lst):axxs[xi].plot(tmp_xsect_x[tmp_datstr],tmp_xsect_mld_dat[tmp_datstr],'k', lw = 0.5)
                                
                                    
                                    #for axi,tmpax  in enumerate(axxs): 
                                    for axi,tmp_datstr in enumerate(Dataset_lst): 
                                        
                                        #tmpax = axxs[xi]
                                        for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].axvline(xi,color = 'k', alpha = 0.5, ls = '--') 
                                        #for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].text(xi,xs_ylim[0] - xs_ylim.ptp()*0.9   ,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                        for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].text(xi,xs_ylim[0] - np.ptp(xs_ylim)*0.9   ,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                        plt.colorbar(paxxs[axi], ax = axxs[axi])
                                    plt.colorbar(paxxs[2], ax = axxs[2])
                                    for xi,tmp_datstr in enumerate(Dataset_lst): set_perc_clim_pcolor_in_region(5,95,ax = axxs[axi])
                                    set_perc_clim_pcolor_in_region(5,95,ax = axxs[2], sym = True)

                                    #xmapax = figxs.add_axes([0.075,0.15,0.2,0.5], frameon=False)
                                    #xmapax = figxs.add_axes([0.075,0.15,0.175,0.4], frameon=False)
                                    xmapax = figxs.add_axes([0.075,0.025,0.175,0.2], frameon=False)
                                    #pdb.set_trace()
                                    xs_pe = [pe.Stroke(linewidth=2, foreground='w'), pe.Normal()]
                                    #pdb.set_trace()
                                    if var_dim[var] == 3:
                                        xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                    elif var_dim[var] == 4:
                                        xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][0][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_dict[xsect_secdataset_proc],xsect_lat_dict[xsect_secdataset_proc],'r-', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    #xmapax.plot(xsect_lon_pnt_mat,xsect_lat_pnt_mat,'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    #xmapax.plot(xsect_lon_pnt_mat[0],xsect_lat_pnt_mat[0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc],xsect_lat_pnt_dict[xsect_secdataset_proc],'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc][0],xsect_lat_pnt_dict[xsect_secdataset_proc][0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.axis('equal')
                                    xmapax.set_xticks([])
                                    xmapax.set_yticks([]) 
                                else:
                                
                                    figxs = plt.figure()
                                    figxs.set_figheight(8)
                                    figxs.set_figwidth(8*1.5)
                                    figxs.suptitle('Cross-section: %s'%nice_varname_dict[var], fontsize = 20)
                                    plt.subplots_adjust(top=0.92,bottom=0.05,left=0.05,right=1,hspace=0.25,wspace=0.6)
                                    axxs = [plt.subplot(211),plt.subplot(212)]
                                    paxxs = []
                                    for xi,tmp_datstr in enumerate(Dataset_lst): paxxs.append(axxs[xi].pcolormesh(tmp_xsect_x[tmp_datstr],tmp_xsect_z[tmp_datstr],tmp_xsect_dat[tmp_datstr]))
                                    for tmpax  in axxs: tmpax.invert_yaxis()
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_title(tmp_datstr)
                                    xs_ylim = np.array(axxs[0].get_ylim())
                                    xs_xlim = np.array(axxs[0].get_xlim())
                                    xs_ylim[0] = tmp_xsect_z[tmp_datstr][~(tmp_xsect_x[tmp_datstr]*tmp_xsect_z[tmp_datstr]*tmp_xsect_dat[tmp_datstr]).mask].max()
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_ylim(xs_ylim)

                                    if do_MLD:
                                        for xi,tmp_datstr in enumerate(Dataset_lst):axxs[xi].plot(tmp_xsect_x[tmp_datstr],tmp_xsect_mld_dat[tmp_datstr],'k', lw = 0.5)
                                
                                
                                    #for axi,tmpax  in enumerate(axxs): 
                                    for axi,tmp_datstr in enumerate(Dataset_lst): 
                                        #tmpax = axxs[axi]
                                        for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].axvline(xi,color = 'k', alpha = 0.5, ls = '--') 
                                        #for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].text(xi,xs_ylim[0] - xs_ylim.ptp()*0.9   ,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                        for xi in xsect_pnt_ind_dict[tmp_datstr]: axxs[axi].text(xi,xs_ylim[0] - np.ptp(xs_ylim)*0.9   ,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                        plt.colorbar(paxxs[axi], ax = axxs[axi])
                                    for xi,tmp_datstr in enumerate(Dataset_lst): set_perc_clim_pcolor_in_region(5,95,ax = axxs[axi])
                                    
                                    #xmapax = figxs.add_axes([0.075,0.15,0.2,0.5], frameon=False)
                                    #xmapax = figxs.add_axes([0.075,0.15,0.175,0.4], frameon=False)
                                    xmapax = figxs.add_axes([0.075,0.025,0.175,0.2], frameon=False)
                                    #pdb.set_trace()
                                    xs_pe = [pe.Stroke(linewidth=2, foreground='w'), pe.Normal()]
                                    #pdb.set_trace()
                                    if var_dim[var] == 3:
                                        xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                    elif var_dim[var] == 4:
                                        xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][0][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_dict[xsect_secdataset_proc],xsect_lat_dict[xsect_secdataset_proc],'r-', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    #xmapax.plot(xsect_lon_pnt_mat,xsect_lat_pnt_mat,'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    #xmapax.plot(xsect_lon_pnt_mat[0],xsect_lat_pnt_mat[0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc],xsect_lat_pnt_dict[xsect_secdataset_proc],'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc][0],xsect_lat_pnt_dict[xsect_secdataset_proc][0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                    xmapax.axis('equal')
                                    xmapax.set_xticks([])
                                    xmapax.set_yticks([]) 

                            #else:if (nDataset == 2)&(var_dim[var] == 4):
                            #elif (nDataset == 1):
                            else:
                                figxs = plt.figure()
                                figxs.set_figheight(4*1.2)
                                figxs.set_figwidth(8*1.5)
                                figxs.suptitle('Cross-section: %s'%nice_varname_dict[var], fontsize = 20)
                                plt.subplots_adjust(top=0.89,bottom=0.1,left=0.05,right=0.975,hspace=0.2,wspace=0.6)
                                axxs = [plt.subplot(111)]
                                paxxs = []
                                if var_dim[var] == 4:
                                    paxxs.append(axxs[0].pcolormesh(tmp_xsect_x[xsect_secdataset_proc],tmp_xsect_z[xsect_secdataset_proc],tmp_xsect_dat[xsect_secdataset_proc]))
                                    axxs[0].invert_yaxis()
                                    plt.colorbar(paxxs[0], ax = axxs[0])
                                    set_perc_clim_pcolor_in_region(5,95,ax = axxs[0])
                                    if do_MLD:
                                        plt.plot(tmp_xsect_x[xsect_secdataset_proc],tmp_xsect_mld_dat[xsect_secdataset_proc],'k', lw = 0.5)
                                
                                elif var_dim[var] == 3:
                                    axxs[0].axhline(0,color = 'k', lw = 0.5) 
                                    
                                    for xi,tmp_datstr in enumerate(Dataset_lst):axxs[0].plot(tmp_xsect_x[tmp_datstr],tmp_xsect_dat[tmp_datstr],label = tmp_datstr)
                                    axxs[0].legend()

                                xs_xlim = np.array(axxs[0].get_xlim())
                                xs_ylim = np.array(axxs[0].get_ylim())
                                
                                if var_dim[var] == 4:
                                    xs_ylim[0] = tmp_xsect_z[xsect_secdataset_proc][~(tmp_xsect_x[xsect_secdataset_proc]*tmp_xsect_z[xsect_secdataset_proc]*tmp_xsect_dat[xsect_secdataset_proc]).mask].max()
                                    axxs[0].set_ylim(xs_ylim)
                                    #pdb.set_trace()

                                for xi in xsect_pnt_ind_dict[xsect_secdataset_proc]:                            axxs[0].axvline(xi,color = 'k', alpha = 0.5, ls = '--') 

                                if var_dim[var] == 4:
                                    #for xi in xsect_pnt_ind_dict[xsect_secdataset_proc]:         axxs[0].text(xi,xs_ylim[0] - xs_ylim.ptp()*0.9,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                    for xi in xsect_pnt_ind_dict[xsect_secdataset_proc]:         axxs[0].text(xi,xs_ylim[0] - np.ptp(xs_ylim)*0.9,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                elif var_dim[var] == 3:
                                    #for xi in xsect_pnt_ind_dict[xsect_secdataset_proc]:         axxs[0].text(xi,xs_ylim[0] + xs_ylim.ptp()*0.9,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                    for xi in xsect_pnt_ind_dict[xsect_secdataset_proc]:         axxs[0].text(xi,xs_ylim[0] + np.ptp(xs_ylim)*0.9,lon_lat_to_str(xsect_lon_dict[tmp_datstr][xi],xsect_lat_dict[tmp_datstr][xi])[0], rotation = 270, ha = 'left', va = 'top')
                                #axxs[0].set_xlim([0,xs_xlim[1]*1.01])
                                #pdb.set_trace()

                                #xmapax = figxs.add_axes([0.075,0.15,0.2,0.5], frameon=False)
                                xmapax = figxs.add_axes([0.075,0.15,0.175,0.4], frameon=False)
                                
                                #pdb.set_trace()
                                xs_pe = [pe.Stroke(linewidth=2, foreground='w'), pe.Normal()]
                                #pdb.set_trace()
                                if var_dim[var] == 3:
                                    xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                elif var_dim[var] == 4:
                                    xsmapconax = xmapax.contour(lon_d[1][1:-1,1:-1],lat_d[1][1:-1,1:-1],data_inst['Dataset 1'][0][1:-1,1:-1].mask, linewidths = 0.5, colors = 'k', path_effect = xs_pe)
                                xmapax.plot(xsect_lon_dict[xsect_secdataset_proc],xsect_lat_dict[xsect_secdataset_proc],'r-', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                #xmapax.plot(xsect_lon_mat,xsect_lat_mat,'r-', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                #xmapax.plot(xsect_lon_pnt_mat,xsect_lat_pnt_mat,'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                #xmapax.plot(xsect_lon_pnt_mat[0],xsect_lat_pnt_mat[0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                #xmapax.plot(xsect_lon_dict[xsect_secdataset_proc],xsect_lat_dict[xsect_secdataset_proc],'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc],xsect_lat_pnt_dict[xsect_secdataset_proc],'k+', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                xmapax.plot(xsect_lon_pnt_dict[xsect_secdataset_proc][0],xsect_lat_pnt_dict[xsect_secdataset_proc][0],'kx', alpha = 0.5, lw = 0.5)#,path_effect = xs_pe)
                                xmapax.axis('equal')
                                xmapax.set_xticks([])
                                xmapax.set_yticks([]) 



                            xs_close_win_meth = 1
                            # 1: closes when you click on it
                            # 2: stays open till you close it

                            if xs_close_win_meth == 1:
                                xclickax = figxs.add_axes([0,0,1,1], frameon=False)
                                xclickax.axis('off')

                            # redraw canvas
                            figxs.canvas.draw()
                            
                            #flush canvas
                            figxs.canvas.flush_events()
                            
                            # Show plot, and set it as the current figure and axis
                            figxs.show()
                            plt.figure(figxs.figure)
                            

                            if xs_close_win_meth == 1:
                                plt.sca(xclickax)
                            
                                ###################################
                                # Close on button press: #JT COBP #
                                ###################################


                                close_xsax = False
                                while close_xsax == False:

                                    # get click location
                                    tmpxsbutloc = plt.ginput(1, timeout = 3) #[(0.3078781362007169, 0.19398809523809524)]
                                            
                                    #pdb.set_trace()
                                    if len(tmpxsbutloc)!=1:
                                        #print('tmpxsbutloc len != 1',tmpxsbutloc )
                                        #close_xsax = True
                                        continue
                                        #pdb.set_trace()
                                    else:
                                        if len(tmpxsbutloc[0])!=2:
                                            close_xsax = True
                                            #print('tmpxsbutloc[0] len != 2',tmpxsbutloc )
                                            continue
                                            #pdb.set_trace()
                                        # was a button clicked?
                                        # if so, record which and allow the window to close
                                        if (tmpxsbutloc[0][0] >= 0) & (tmpxsbutloc[0][0] <= 1) & (tmpxsbutloc[0][1] >= 0) & (tmpxsbutloc[0][1] <= 1):
                                            #pdb.set_trace()
                                            close_xsax = True

                                    # quit of option box is closed without button press.
                                    if plt.fignum_exists(figxs) == False:
                                        close_xsax = True
                                        
                                
                                # close figure
                                if close_xsax:
                                    if figxs is not None:
                                        if plt.fignum_exists(figxs.number):
                                            plt.close(figxs)
                                ##import time            
                                #figxs_exists = plt.fignum_exists(figxs.number)
                                #while figxs_exists:
                                #    time.sleep(0.5)
                                #    figxs_exists = plt.fignum_exists(figxs.number)

                            
                            for xs_map_ax in xs_map_ax_lst:
                                rem_loc = xs_map_ax.pop(0)
                                rem_loc.remove()
                                

                            figxs.canvas.draw()
                            figxs.canvas.flush_events()
                            #plt.figure(fig.figure)
                            #plt.sca(clickax)


                        elif but_name == 'TS Diag':
                            if figts is not None:
                                if plt.fignum_exists(figts.number):
                                    plt.close(figts)
                                

                            #pdb.set_trace()
                            if button_press:
                            #if ts_diag_coord.mask.all() 
                            #    if ((ts_diag_coord == np.ma.array([ii,jj,ti])).all() == False):
                                secondary_fig = True

                                ## TS diagram Pycnoclines
                                tmp_t_arr = np.arange(0,30,.1)
                                tmp_s_arr = np.arange(15,40,.1)

                                tmp_s_mat,tmp_t_mat = np.meshgrid(tmp_s_arr,tmp_t_arr)
                                tmp_rho_mat = sw_dens(tmp_t_mat,tmp_s_mat)



                                
                                tmp_T_data = {}
                                tmp_S_data = {}
                                tmp_gdept = {}
                                tmp_mld1 = {}
                                tmp_mld2 = {}
                                tmp_sigma_density_data = {}

                                for tmp_datstr in Dataset_lst:

                                    tmp_T_data[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
                                    tmp_S_data[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
                                    tmp_gdept[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
                                    tmp_mld1[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
                                    tmp_mld2[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked
                                    tmp_sigma_density_data[tmp_datstr] = np.ma.zeros((nz))*np.ma.masked


                                    th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])

                                    
                                    if (configd[th_d_ind] == configd[1])| (tmp_datstr== Dataset_lst[0]):
                                        tmp_T_data[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj,ii].load())
                                        tmp_S_data[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj,ii].load())
                                        tmp_gdept[tmp_datstr] = np.array(grid_dict[tmp_datstr]['gdept'])[:,jj,ii]
                                        tmp_mld1[tmp_datstr] = np.ma.masked
                                        tmp_mld2[tmp_datstr] = np.ma.masked
                                        if 'mld25h_1' in var_d[th_d_ind]['mat']: tmp_mld1[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['mld25h_1'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][jj,ii].load())
                                        if 'mld25h_2' in var_d[th_d_ind]['mat']: tmp_mld2[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['mld25h_2'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][jj,ii].load())

                                    else:
                                        if 'votemper' in var_d[th_d_ind]['mat']:tmp_T_data[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                        if 'vosaline' in var_d[th_d_ind]['mat']:tmp_S_data[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                        if 'mld25h_1' in var_d[th_d_ind]['mat']:tmp_mld1[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['mld25h_1'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                        if 'mld25h_2' in var_d[th_d_ind]['mat']:tmp_mld2[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr]['T'][ldi].variables['mld25h_2'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                        tmp_gdept[tmp_datstr] =  np.array(grid_dict[tmp_datstr]['gdept'])[:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']]               

                                    
                                    tmp_sigma_density_data[tmp_datstr] = sw_dens(tmp_T_data[tmp_datstr],tmp_S_data[tmp_datstr])-1000.
                            
                                figts = plt.figure()
                                figts.set_figheight(8*1.2)
                                figts.set_figwidth(6*1.5)
                                axsp = figts.add_axes([0.1, 0.10, 0.3,  0.75])
                                axts = figts.add_axes([0.5, 0.55, 0.4,  0.30])
                                plt.subplots_adjust(top=0.8,bottom=0.11,left=0.125,right=0.9,hspace=0.2,wspace=0.6)
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axsp.plot(tmp_S_data[tmp_datstr],tmp_gdept[tmp_datstr],color = 'g', linestyle = linestyle_str[dsi])
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axsp.axhline(tmp_mld1[tmp_datstr], color = '0.5', linestyle = linestyle_str[dsi])
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axsp.axhline(tmp_mld2[tmp_datstr], color = '0.25', linestyle = linestyle_str[dsi])
                                axsp.spines['bottom'].set_color('g')
                                axsp.spines['top'].set_visible(False)
                                axsp.set_xlabel('Salinity')  
                                axsp.xaxis.label.set_color('g')
                                axsp.tick_params(axis = 'x',colors = 'g')
                                axsp.invert_yaxis()
                                #
                                axtp = axsp.twiny()
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axtp.plot(tmp_T_data[tmp_datstr],tmp_gdept[tmp_datstr],color = 'r',  linestyle = linestyle_str[dsi])
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axtp.plot(np.ma.masked,color = 'k', label =  fig_lab_d[tmp_datstr], linestyle = linestyle_str[dsi])
                                plt.legend(loc = 'lower left', fancybox=True, framealpha=0.75)
                                axtp.set_xlabel('Temperature')
                                axtp.spines['top'].set_color('r')
                                axtp.tick_params(axis = 'x',colors = 'r')
                                axtp.spines['bottom'].set_visible(False)
                                axtp.xaxis.label.set_color('r')
                                axrp = axsp.twiny()
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axrp.plot(tmp_sigma_density_data[tmp_datstr],tmp_gdept[tmp_datstr],color = 'b', lw = 0.5, linestyle = linestyle_str[dsi])
                                axrp.set_xlabel('Density')
                                axrp.spines['top'].set_color('b')
                                axrp.tick_params(axis = 'x',colors = 'b')
                                axrp.spines['bottom'].set_visible(False)
                                axrp.xaxis.label.set_color('b')
                                axrp.spines['top'].set_position(('axes', 1.1))
                                #
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axts.plot(tmp_S_data[tmp_datstr],tmp_T_data[tmp_datstr],color = 'b', linestyle = linestyle_str[dsi])
                                axts.set_xlabel('Salinity')
                                axts.set_ylabel('Temperature')
                                tmprhoxlim = axts.get_xlim()
                                tmprhoylim = axts.get_ylim()
                                axts.contour(tmp_s_mat,tmp_t_mat,tmp_rho_mat, np.arange(0,50,0.1), colors = 'k', linewidths = 0.5, alphas = 0.5, linestyles = '--')
                                axts.set_xlim(tmprhoxlim)
                                axts.set_ylim(tmprhoylim)
                                figts_lab_str = '%s\n\n%s'%(lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0],time_datetime[ti])
                                #for dsi,tmp_datstr in enumerate(Dataset_lst): figts_lab_str = figts_lab_str + '\n\n%s'%fig_lab_d[tmp_datstr]
                                #plt.text(0.5, 0.1, figts_lab_str, fontsize=14, transform=figts.transFigure, ha = 'left', va = 'bottom')
                                plt.text(0.5, 0.9, figts_lab_str, fontsize=14, transform=figts.transFigure, ha = 'left', va = 'bottom')

                                
                                TSfig_out_name = '%s/output_TSDiag_%s_%s_%s'%(fig_dir,fig_lab,lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[3],time_datetime[ti].strftime('%Y%m%dT%H%MZ'))
                                
                                figts.savefig(TSfig_out_name + '.png')
                                figts.show()



                                #except:
                                #    print('TS Diag error')
                                #    pdb.set_trace()
                        elif but_name == 'Clim: Zoom': 
                            if mouse_info['button'].name == 'MIDDLE':
                                clim = None
                            elif mouse_info['button'].name == 'LEFT':


                                plt.sca(clickax)
                    
                                func_but_text_han['Clim: Zoom'].set_color('r')
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()
                                tmpczoom = plt.ginput(2)
                                clim = np.array([tmpczoom[0][1],tmpczoom[1][1]])
                                clim.sort()

                                func_but_text_han['Clim: Zoom'].set_color('k')
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()

                            elif mouse_info['button'].name == 'RIGHT':
                                clim = np.array(get_clim_pcolor(ax = ax[0]))
                                if climnorm is None:
                                    #clim = np.array([clim.mean() - clim.ptp(),clim.mean() + clim.ptp()])
                                    clim = np.array([clim.mean() - np.ptp(clim),clim.mean() + np.ptp(clim)])
                                else:
                                    #clim = np.log10(np.array([(10**clim).mean() - (10**clim).ptp(),(10**clim).mean() + (10**clim).ptp()]))
                                    clim = np.log10(np.array([(10**clim).mean() - np.ptp((10**clim)),(10**clim).mean() + np.ptp((10**clim))]))
                        
                            '''

                        elif but_name == 'Clim: Expand': 
                            clim = np.array(get_clim_pcolor(ax = ax[0]))
                            if climnorm is None:
                                clim = np.array([clim.mean() - clim.ptp(),clim.mean() + clim.ptp()])
                            else:
                                clim = np.log10(np.array([(10**clim).mean() - (10**clim).ptp(),(10**clim).mean() + (10**clim).ptp()]))
                            
                            '''

                        elif but_name == 'LD time':
                            ldi+=1
                            if ldi == nldi: ldi = 0
                            func_but_text_han['LD time'].set_text('LD time: %s'%ld_lab_mat[ldi])
                            reload_map = True
                            reload_ew = True
                            reload_ns = True
                            reload_hov = True
                            reload_ts = True



                        elif but_name == 'Fcst Diag':
                            if figfc is not None:
                                if plt.fignum_exists(figfc.number):
                                    plt.close(figfc)
                                

                            #pdb.set_trace()
                            if button_press:
                                secondary_fig = True

                                fsct_hov_dat_dict = {}
                                fsct_ts_dat_dict = {}
                                for tmp_datstr in Dataset_lst:fsct_hov_dat_dict[tmp_datstr] = np.ma.zeros(((nldi,)+hov_dat_dict[tmp_datstr].shape))*np.ma.masked 
                                for tmp_datstr in Dataset_lst:fsct_ts_dat_dict[tmp_datstr] = np.ma.zeros(((nldi,)+ts_dat_dict[tmp_datstr].shape))*np.ma.masked 

                                fsct_hov_x = np.ma.zeros((nldi,)+hov_dat_dict['x'].shape, dtype = 'object')*np.ma.masked
                                fsct_ts_x = np.ma.zeros((nldi,)+ts_dat_dict['x'].shape, dtype = 'object')*np.ma.masked


                                try:
                                    ld_time_offset = [int(ss) for ss in ld_lab_mat]
                                except:
                                    ld_time_offset = [int(ss*24 - 36) for ss in range(nldi)]

                                fcdata_start = datetime.now()
                                print('Extracting forecast data:',fcdata_start)


                                for fcst_ldi in range(nldi):

                                    fsct_hov_dat = reload_hov_data_comb(var,var_d[1]['mat'],var_grid['Dataset 1'],var_d['d'],fcst_ldi, thd,time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,xarr_dict, load_second_files,Dataset_lst,configd)
                                    for tmp_datstr in Dataset_lst:fsct_hov_dat_dict[tmp_datstr][fcst_ldi] = fsct_hov_dat[tmp_datstr]
                                    fsct_hov_x[fcst_ldi] = fsct_hov_dat['x'] + timedelta(hours = ld_time_offset[fcst_ldi])
                
                                    fsct_ts_dat = reload_ts_data_comb(var,var_dim,var_grid['Dataset 1'],ii,jj,iijj_ind,fcst_ldi,fsct_hov_dat,time_datetime,z_meth,zz,zi,xarr_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,load_second_files)
                                    
                                    for tmp_datstr in Dataset_lst:fsct_ts_dat_dict[tmp_datstr][fcst_ldi] = fsct_ts_dat[tmp_datstr]
                                    fsct_ts_x[fcst_ldi] = fsct_ts_dat['x'] + timedelta(hours = ld_time_offset[fcst_ldi])
                                print('Extracted forecast data:',datetime.now(), datetime.now() - fcdata_start)


                                

                                    
                                figfc_lab_str = '%s forecast diagram for \n%s'%(nice_varname_dict[var],lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0])


                                if var_dim[var] == 4:  
                                    figfc_lab_str = '%s (%s) forecast diagram\nfor %s'%(nice_varname_dict[var],nice_lev,lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0])
                                elif var_dim[var] == 3:
                                    figfc_lab_str = '%s forecast diagram\nfor %s'%(nice_varname_dict[var],lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0])



                                figfc = plt.figure()
                                figfc.set_figheight(5)
                                figfc.set_figwidth(6)
                                figfc.suptitle(figfc_lab_str, fontsize = 16) 
                                axfc = []
                                if load_second_files:                       
                                    figfc.set_figheight(8)    
                                    axfc.append(plt.subplot(2,1,1))  
                                    axfc.append(plt.subplot(2,1,2))
                                    plt.subplots_adjust(top=0.875,bottom=0.11,left=0.125,right=0.9,hspace=0.2,wspace=0.6) 
                                else:
                                    plt.subplots_adjust(top=0.825,bottom=0.11,left=0.125,right=0.9,hspace=0.2,wspace=0.6) 
                                    axfc.append(plt.subplot(1,1,1)) 


                                axfc[0].plot(fsct_ts_x,fsct_ts_dat_dict['Dataset 1'][:,:], '0.5' )                   
                                axfc[0].plot(fsct_ts_x[0,:],fsct_ts_dat_dict['Dataset 1'][0,:],'ro' )            
                                axfc[0].plot(fsct_ts_x[-1,:],fsct_ts_dat_dict['Dataset 1'][-1,:],'x', color = '0.5')
                                axfc[0].set_title(fig_lab_d['Dataset 1'])
                                if load_second_files:       
                                    axfc[1].plot(fsct_ts_x,fsct_ts_dat_dict['Dataset 2'][:,:], '0.5' )                   
                                    axfc[1].plot(fsct_ts_x[0,:],fsct_ts_dat_dict['Dataset 2'][0,:],'ro' )
                                    axfc[1].plot(fsct_ts_x[-1,:],fsct_ts_dat_dict['Dataset 2'][-1,:],'x', color = '0.5')
                                    axfc[1].set_title(fig_lab_d['Dataset 2'])
                                figfc.show()

                            #pdb.set_trace()

                        elif but_name == 'Clim: pair':
                            if clim_pair:
                                func_but_text_han['Clim: pair'].set_color('k')
                                clim_pair = False
                            else:
                                func_but_text_han['Clim: pair'].set_color('gold')
                                clim_pair = True

                        elif but_name == 'Clim: sym':
                            if clim_sym_but == 0:
                                func_but_text_han['Clim: sym'].set_color('r')
                                #curr_cmap = scnd_cmap
                                clim_sym_but = 1
                                #clim_sym_but_norm_val = clim_sym
                                clim_sym = True
                                
                            elif clim_sym_but == 1:
                                func_but_text_han['Clim: sym'].set_color('k')
                                clim_sym_but = 0
                                
                                
                                #curr_cmap = base_cmap
                                #func_but_text_han['ColScl'].set_text('Col: Linear')
                                #col_scl = 0
                                #clim_sym = clim_sym_but_norm_val
                                

                        elif but_name == 'Hov/Time':
                            if hov_time:
                                func_but_text_han['Hov/Time'].set_color('0.5')
                                hov_time = False
                            else:
                                func_but_text_han['Hov/Time'].set_color('darkgreen')
                                hov_time = True
                                reload_hov = True
                                reload_ts = True

                        elif but_name == 'Show Prof':
                            if profvis:
                                func_but_text_han['Show Prof'].set_color('0.5')
                                profvis = False
                                for tmpax, tmppos in zip(ax,ax_position_dims): tmpax.set_position(tmppos)
                                ax[-1].set_visible(profvis)

                            else:
                                func_but_text_han['Show Prof'].set_color('darkgreen')
                                profvis = True
                                for tmpax, tmppos in zip(ax,ax_position_dims_prof): tmpax.set_position(tmppos)
                                ax[-1].set_visible(profvis)


                        elif but_name == 'regrid_meth':
                            if regrid_meth == 1:
                                func_but_text_han['regrid_meth'].set_text('Regrid: Bilin')
                                regrid_meth = 2
                                reload_map = True
                                reload_ew = True
                                reload_ns = True
                            elif regrid_meth == 2:
                                func_but_text_han['regrid_meth'].set_text('Regrid: NN')
                                regrid_meth = 1
                                reload_map = True
                                reload_ew = True
                                reload_ns = True

                        elif but_name == 'Contours':
                            if do_cont:
                                func_but_text_han['Contours'].set_color('k')
                                do_cont = False
                            else:
                                func_but_text_han['Contours'].set_color('darkgreen')
                                do_cont = True


                        elif but_name == 'Grad':
                            if mouse_info['button'].name == 'LEFT':
                                if do_grad == 0:
                                    do_grad = 1
                                elif do_grad == 1:
                                    do_grad = 2
                                elif do_grad == 2:
                                    do_grad = 0
                            if mouse_info['button'].name == 'RIGHT':
                                if do_grad == 0:
                                    do_grad = 2
                                elif do_grad == 1:
                                    do_grad = 0
                                elif do_grad == 2:
                                    do_grad = 1

                            if do_grad == 0:
                                func_but_text_han['Grad'].set_color('0.5')
                                func_but_text_han['Grad'].set_text('Grad')
                            elif do_grad == 1:
                                func_but_text_han['Grad'].set_color('darkgreen')
                                func_but_text_han['Grad'].set_text('Horiz Grad')
                            elif do_grad == 2:
                                func_but_text_han['Grad'].set_color('gold')
                                func_but_text_han['Grad'].set_text('Vert Grad')

                            reload_map = True
                            reload_ew = True
                            reload_ns = True
                            reload_hov = True
                            reload_ts = True
                            '''
                            if do_grad == 0:
                                func_but_text_han['Grad'].set_color('darkgreen')
                                func_but_text_han['Grad'].set_text('Horiz Grad')
                                do_grad = 1
                                reload_map = True
                                reload_ew = True
                                reload_ns = True
                                reload_hov = True
                                reload_ts = True
                            elif do_grad == 1:
                                func_but_text_han['Grad'].set_color('gold')
                                func_but_text_han['Grad'].set_text('Vert Grad')

                                do_grad = 2
                                reload_map = True
                                reload_ew = True
                                reload_ns = True
                                reload_hov = True
                                reload_ts = True
                            elif do_grad == 2:
                                func_but_text_han['Grad'].set_color('0.5')
                                func_but_text_han['Grad'].set_text('Grad')

                                do_grad = 0
                                reload_map = True
                                reload_ew = True
                                reload_ns = True
                                reload_hov = True
                                reload_ts = True
                            '''
    

                        elif but_name == 'Vis curr':

                        
                            if vis_curr == 1:
                                vis_curr = 0
                                reload_UV_map = False
                                func_but_text_han['Vis curr'].set_color('k')
                            else:
                                vis_curr = 1
                                reload_UV_map = True
                                func_but_text_han['Vis curr'].set_color('darkgreen')

                        elif but_name == 'Sec Grid':
                            if Sec_regrid:
                                    Sec_regrid = False
                                    func_but_text_han['Sec Grid'].set_color('k')
                            else:
                                Sec_regrid = True
                                func_but_text_han['Sec Grid'].set_color('darkgreen')
                                if 'Dataset 2_Sec_Grid' not in map_dat_dict.keys():
                                    reload_map = True
                                    reload_UV_map = True
                        elif but_name == 'Time Diff':

                            if ti == 0:
                                func_but_text_han['Time Diff'].set_color('0.5')
                            else:
                                if Time_Diff:
                                    Time_Diff = False
                                    func_but_text_han['Time Diff'].set_color('k')
                                else:
                                    Time_Diff = True
                                    func_but_text_han['Time Diff'].set_color('darkgreen')


                        elif but_name == 'ColScl':
                            if secdataset_proc in Dataset_lst:
                                if col_scl == 0:
                                    func_but_text_han['ColScl'].set_text('Col: High')
                                    col_scl = 1
                                    curr_cmap = base_cmap_high
                                elif col_scl == 1:
                                    func_but_text_han['ColScl'].set_text('Col: Low')
                                    curr_cmap = base_cmap_low
                                    col_scl = 2
                                elif col_scl == 2:
                                    func_but_text_han['ColScl'].set_text('Col: Linear')
                                    curr_cmap = base_cmap
                                    col_scl = 0
                            else:
                                curr_cmap = scnd_cmap


                        
                        elif but_name in secdataset_proc_list:
                            secdataset_proc = but_name


                            for tmpsecdataset_proc in secdataset_proc_list: func_but_text_han[tmpsecdataset_proc].set_color('k')


                            func_but_text_han[but_name].set_color('darkgreen')

                            if do_ensemble:
                                Ens_stat = None
                                for ens_stat in ens_stat_lst: func_but_text_han[ens_stat].set_color('k')
                                

                        #elif do_ensemble:
                        #    if but_name in ens_stat_lst:
                        elif but_name in ens_stat_lst:
                            #if do_ensemble:
                        
                            Ens_stat = but_name
                            for tmpsecdataset_proc in secdataset_proc_list + ens_stat_lst: func_but_text_han[tmpsecdataset_proc].set_color('k')
                            func_but_text_han[but_name].set_color('darkgreen')

                            if Ens_stat not in ['EnsMean']:
                                clim_pair = False
                                func_but_text_han['Clim: pair'].set_color('k')
                                


                        elif but_name in ['Surface','Near-Bed','Surface-Bed','Depth-Mean']:
                            if var_dim[var] == 4:
                                
                                if but_name == 'Surface':z_meth = 'ss'
                                if but_name == 'Near-Bed': z_meth = 'nb'
                                if but_name == 'Surface-Bed': z_meth = 'df'
                                if but_name == 'Depth-Mean': z_meth = 'zm'
                                reload_map = True
                                reload_ts = True

                                func_but_text_han['Depth level'].set_color('k')
                                func_but_text_han['Surface'].set_color('k')
                                func_but_text_han['Near-Bed'].set_color('k')
                                func_but_text_han['Surface-Bed'].set_color('k')
                                func_but_text_han['Depth-Mean'].set_color('k')
                                func_but_text_han[but_name].set_color('r')
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()

                        elif but_name in ['Depth level']:
                            func_but_text_han['Depth level'].set_color('k')
                            func_but_text_han['Surface'].set_color('k')
                            func_but_text_han['Near-Bed'].set_color('k')
                            func_but_text_han['Surface-Bed'].set_color('k')
                            func_but_text_han['Depth-Mean'].set_color('k')
                            func_but_text_han[but_name].set_color('r')
                            z_meth = z_meth_default    
                            reload_map = True
                            reload_ts = True
                        elif but_name in ['Save Figure']:                        
                            save_figure_funct()

                        elif but_name in mode_name_lst:
                            if mode == 'Loop': 
                                mouse_in_Click = False
                            mode = but_name
                            func_but_text_han['Click'].set_color('k')
                            func_but_text_han['Loop'].set_color('k')
                            func_but_text_han[mode].set_color('gold')
                            reload_map = True
                            reload_ew = True
                            reload_ns = True
                            reload_hov = False
                            reload_ts = False
                            if do_Obs:
                                reload_Obs = True
                        elif but_name in 'Quit':
                            print('Closing')
                            print('')
                            print('')
                            print('')
                            return
                        else:
                            print('but_name:',but_name)
                            print('No function for but_name')
                            pdb.set_trace()
                        if verbose_debugging: print('clim:',clim)
                            
                            

            if do_timer: timer_lst.append(('Finished var func',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            
            plt.sca(ax[0])
                    
            
            ###################################################################################################
            ### remove contours, colorbars, images, lines, text, ready for next cycle
            ###################################################################################################
            if do_timer: timer_lst.append(('Remove contours, images, lines text',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)
            
            
            if verbose_debugging: print('Interpret Mouse click: remove lines and axes', datetime.now())
            #pdb.set_trace()
            #print(ii,jj, ti, zz,var)
            print("selected ii = %i,jj = %i,ti = %i,zz = %i, var = '%s'"%(ii,jj, ti, zz,var))
            # after selected indices and vareiabels, delete plots, ready for next cycle
            #pdb.set_trace()
            for tmp_cax in cax:tmp_cax.remove()
            '''
            print(len(cax))
            for tmp_cax in cax:
                print(tmp_cax)
                type(tmp_cax)
                try:
                    tmp_cax.remove()
                except:
                    pdb.set_trace()
            '''
            del(cax)
            

            for tmp_pax in pax:tmp_pax.remove()
            for tmp_cs_line in cs_line:tmp_cs_line.remove()
            if var_dim[var] == 3:
                for tmppax2d in pax2d:
                    rem_loc = tmppax2d.pop(0)
                    rem_loc.remove()

            for tsax in tsax_lst:
                rem_loc = tsax.pop(0)
                rem_loc.remove()

            for pfax in pfax_lst:
                rem_loc = pfax.pop(0)
                rem_loc.remove()
            if do_MLD:
                #pdb.set_trace()
                # remove profile
                for mldax in mldax_lst:
                    rem_loc = mldax.pop(0)
                    rem_loc.remove()
                
            if do_Obs:
                # remove profile
                for opax in opax_lst:
                    rem_loc = opax.pop(0)
                    rem_loc.remove()

                # remove selected observation point
                for oxax in oxax_lst:
                    rem_loc = oxax.pop(0)
                    rem_loc.remove()

                # remove scatter point
                for oax in oax_lst:oax.remove()
                #pdb.set_trace()
                for opaxtx in opaxtx_lst:opaxtx.remove()
                #pdb.set_trace()
                #opaxtx.remove()

                
            # remove vectors before next iteration
            for tmpvisax in visax:
                rem_loc = tmpvisax.pop(0)
                rem_loc.remove()
                
            # remove contour before next iteration
            '''
            old method stopped working 6/6/2025
            for tmpconax in conax:
                for tmpconaxcoll in tmpconax.collections: tmpconaxcoll.remove()
            '''            

            for tmpconax in conax: tmpconax.remove()
                
                
            '''
            for tsax in tsaxtx_lst:
                rem_loc = tsax.pop(0)
                rem_loc.remove()

            for tsax in tsaxtxd_lst:
                rem_loc = tsax.pop(0)
                rem_loc.remove()
            '''

            #rem_loc2 = tsax2.pop(0)
            #rem_loc2.remove()


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
            
            if verbose_debugging: print('Cycle', datetime.now())

            if do_timer: timer_lst.append(('Cycle ended',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)


def main():
    legacy_mode = False

    nemo_slice_zlev_helptext=textwrap.dedent('''\
    Interactive NEMO ncfile viewer.
    ===============================
    Developed by Jonathan Tinker Met Office, UK, December 2023
    ==========================================================
    
    When calling from the command line, it uses a mix of positional values, and keyword value pairs, via argparse.

    The first two positional keywords are the NEMO configuration "config", 
    and the second is the list of input file names "fname_lst" - this is the for the first 
    data set, for the T grid other file list are provided with the --files argument - see later.
    
    config: should be AMM7, AMM15, CO9p2, ORCA025, ORCA12. Other configurations will be supported soon. 
    fname_lst: supports wild cards, but should be  enclosed in quotes.
    e.g.
    python NEMO_nc_slevel_viewer_dev.py amm15 "/directory/to/some/files/prodm_op_am-dm.gridT*-36.nc" 

    if using a variable in the file list use:


    fig_fname_lab=dataset1
    fig_fname_lab_2nd=dataset1


    flist1=$(echo "/directory/to/some/files/${fig_fname_lab}/prodm_op_am-dm.gridT*-36.nc)
    flist2=$(echo "/directory/to/some/files/${fig_fname_lab_2nd}/prodm_op_am-dm.gridT*-36.nc)

    Optional arguments are give as keyword value pairs, with the keyword following a double hypen.
    for some argements, (th, files etc.), there are mulitple values given                                         
    We will list the most useful options first.

    
    --fnames - addition file list for differnt datasets and grid, e.g. to show the different 
        between two sets of files. 
        
        --files is followed by the data set number, and the grid, before the file list,  
        enclose in quotes. Make sure this has the same number of files, with the same dates as 
        fname_lst. This will be checked in later upgrades, but will currently fail if the files
        are inconsistent

    
    --fnames 1 U - specify a consistent set of U and V files, to calculate a drived variable current magintude. 
        assumes the variable vozocrtx is present. Later upgrade will allow the plotting of vectors, 
        and to handle other current variable names. Must have both U_fname_lst and V_fname_lst.
    --fnames 1 V - specify a consistent set of U and V files, to calculate a drived variable current magintude. 
        assumes the variable vomecrty is present. Later upgrade will allow the plotting of vectors, 
        and to handle other current variable names. Must have both U_fname_lst and V_fname_lst.
        
    --fnames 2 U as above for a second data set
    --fnames 2 V as above for a second data set
                                             
    --fnames 1 WW3 for WW3 data on the AMM15 grid, in netcdf files
    --fnames 2 WW3 as above for a second data set
                                             
    --fnames 1 I for NEMOVAR incremnt data 
    --fnames 2 I as above for a second data set
                                             
    --configs - it is now possible to compare two differnt amm7 and amm15 data
    the positional config keyword is enters as --config 1 
    secondary configs are entered as:
    --configs 2
    available configs include:
        AMM7, AMM15, CO9P2, ORCA025, ORCA025EXT or ORCA12                                         
                                             
    --ii            initial ii value
    --jj            initial jj value
    --ti            initial ti value
    --zz            initial zz value
    --lon           initial lon value
    --lat           initial lat value
    --date_ind      initial date value in '%Y%m%d' format, or a differnt format with --date_fmt
    --date_fmt      format for reading dates
    
    When displaying large datasets it can take a long time to load the file (connecting with xarray vis open_mfdataset). 
    When a button press requires new data to be display that can also take time. The slowest part to read new data is 
    when loading time series of data through files - uses in the hovmuller plots and the time series. There is a button
    to turn on and off reloading of these data, which can speed up the response. To speed up the initial display, 
    this can also be turned off at the command line with:
    
    --hov_time False

    --zlim_max - maximum depth to show, often set to 200. Default is None

    Data Thinning
    =============
    To speed up handling of large files, you can "thin" the data, only loading every x row and column of the data:
        data[thd[1]['y0']:thd[1]['y1']:thd[1]['dy'],thd[1]['x0']:thd[1]['x1']:thd[1]['dx']]

    When commparing two data sets, you can thin them separately

    You can also thin how many files are read in, skipping files, and setting the first and last file in the file list,
    
    The argument is --th, followed by the dataset number (1, 2, 3...), the thinning option, and the value

    the first thinning option for the first data set are applied to all datasets, and then replaced if present.

    use the option --th 1 xy 5, --th xy 5, --th 1 df 5

    e.g.
    
    --th 1 dxy 5 thin the data, to only load the 5th row and column
    --th 1 dx 5  thin the data, to only load the 5th row
    --th 1 dy 5  thin the data, to only load the 5th column
                                           
    --th 2 dx 5  thin the data of the second data set, if of a differnt configuration.
    
    or thinned temporally, skipping some of the files: 
    
    --th 1 df    thin the data, to only load the xth file
    --th 1 f0    thin the data, to only load the files after the xth
    --th 1 f1    thin the data, to only load the files before the xth
    
    It is also possible to only load a reduced region:
    
    --th 1 x0    first row to load
    --th 1 x1    last row to load
    --th 1 y0    first column to load
    --th 1 y1    last column to load
    
    --th 2 x0    first row to load of the second data set, if of a differnt configuration.
    --th 2 x1    last row to load of the second data set, if of a differnt configuration.
    --th 2 y0    first column to load of the second data set, if of a differnt configuration.
    --th 2 y1    last column to load of the second data set, if of a differnt configuration.
                                             
    The th arguemnts replace the following legacy argmuents
        thin, thin_x0, thin_x1, thin_files thin_files_0, thin_file_1
        thin_2nd, thin_x0_2nd, thin_x1_2nd, thin_files_2nd thin_files_0_2nd, thin_file_1_2nd
    
    It is possible to save figures, these will also have text files with the settings to recreate the figure
    at a higher resolution (more files, less thining) with just plot

    --fig_dir - directory for figure output
    --fig_lab - label to add to filesnames, so can compare runs.
    --fig_cutout - save full screen, or cut off the buttons - this is the defaulted to True

    --clim_sym use a symetrical colourbar -defaulted to False
    --clim_pair use the same color limits between datasets. Can be changed with a button click
    --use_cmocean - use cmocean colormaps -defaulted to False

    --verbose_debugging - prints out lots of statements at run time, to help debug -defaulted to False

    
    Overview
    ========
    When the viewer loads, there is a series of variable buttons on the left hand side, fuction buttons on the right hand side, and subplots.
    The main subplot is on the left, which is  2d lon lat surface plot. The right hand plots (top to bottom) show a zonal and meridonial slice, 
    a hovmoller plot and a timeseries. The active location is showns as crosshairs on each subplot. 
                                             
    When you click, text will appear in the top right hand saying waiting, which will then update with the function/variable name.
    this shows that it is working.  When you click zoom, the zoom button with turn red after the first click, and then back to 
    black after the second click. 

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
    You can load two data sets using --fname_lst_2nd, and then switch between the dataset, and show there differnce with the "Dataset 1", "Dataset 2", "Dat1-Dat2", "Dat2-Dat1" buttons.

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

    Yo 

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
    


    Saving figures
    ==============
    You can take snap shots of the screen by clicking Save Figure, and then clicking white space. 
    Files will be saved in the dirertory given with the --fig_dir option.
    Figures will be named based on the variable, ii,jj, ti and zz location, and with a figure label
    given with the --fig_lab option. By default, the savedfigure will exclude the buttons. If you want
    the full screen (or the cut out is not optimised) use  the --fig_cutout False option.

    Just plotting
    =============
    When analysing large datasets, the loading and interactivity can be slow. thinnig the data allows 
    reasonable performance at reduce resolution. One approach is to use this low-res option to find 
    intersting features, then save the figure and the options in an text files. These can then be edited
    (e.g. reducing the thinning), and the viewer can be run in "justplot" mode, where it loads the data
    and saves the figures without any interactivity. This can even be run on spice.
    
    --justplot True                 Just plot mode
    --justplot_date_ind             additional dates to plot
    --justplot_secdataset_proc      datasets to plot
    --justplot_z_meth_zz            depths to plot

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



        parser.add_argument('config', type=str, help="AMM7, AMM15, CO9P2, ORCA025, ORCA025EXT or ORCA12")# Parse the argument
        parser.add_argument('fname_lst', type=str, help='Input file list, enclose in "" more than simple wild card')

        parser.add_argument('--zlim_max', type=int, required=False)
        parser.add_argument('--var', type=str)# Parse the argument
        if legacy_mode:
            parser.add_argument('--fname_lst_2nd', type=str, required=False, help='Input file list, enclose in "" more than simple wild card, Check this has the same number of files as the fname_lst')
            parser.add_argument('--config_2nd', type=str, required=False, help="Only AMM7, AMM15. No implemented CO9P2, ORCA025, ORCA025EXT or ORCA12")# Parse the argument

            parser.add_argument('--U_fname_lst', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vozocrtx, enclose in "" more than simple wild card')
            parser.add_argument('--V_fname_lst', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vomecrty, enclose in "" more than simple wild card')
            parser.add_argument('--U_fname_lst_2nd', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vozocrtx, enclose in "" more than simple wild card')
            parser.add_argument('--V_fname_lst_2nd', type=str, required=False, help='Input U file list for current magnitude. Assumes file contains vomecrty, enclose in "" more than simple wild card')

            parser.add_argument('--WW3_fname_lst', type=str, required=False, help='Input WW3 file list for current magnitude. Assumes file contains vozocrtx, enclose in "" more than simple wild card')
            parser.add_argument('--WW3_fname_lst_2nd', type=str, required=False, help='Input WW3 file list for current magnitude. Assumes file contains vozocrtx, enclose in "" more than simple wild card')
        
        parser.add_argument('--preload_data', type=str, required=False)
        parser.add_argument('--allow_diff_time', type=str, required=False)


        if legacy_mode:
            parser.add_argument('--thin', type=int, required=False)
            parser.add_argument('--thin_2nd', type=int, required=False)

            parser.add_argument('--thin_x0', type=int, required=False)
            parser.add_argument('--thin_x1', type=int, required=False)
            parser.add_argument('--thin_y0', type=int, required=False)
            parser.add_argument('--thin_y1', type=int, required=False)
            parser.add_argument('--thin_x0_2nd', type=int, required=False)
            parser.add_argument('--thin_x1_2nd', type=int, required=False)
            parser.add_argument('--thin_y0_2nd', type=int, required=False)
            parser.add_argument('--thin_y1_2nd', type=int, required=False)

            parser.add_argument('--thin_files', type=int, required=False)
            parser.add_argument('--thin_files_0', type=int, required=False)
            parser.add_argument('--thin_files_1', type=int, required=False)

    

        parser.add_argument('--xlim', type=float, required=False, nargs = 2)
        parser.add_argument('--ylim', type=float, required=False, nargs = 2)
        #parser.add_argument('--tlim', type=str, required=False)
        parser.add_argument('--clim', type=float, required=False, nargs = 8)

        parser.add_argument('--ii', type=int, required=False)
        parser.add_argument('--jj', type=int, required=False)
        parser.add_argument('--ti', type=int, required=False)
        parser.add_argument('--zz', type=int, required=False)

        parser.add_argument('--lon', type=float, required=False)
        parser.add_argument('--lat', type=float, required=False)
        parser.add_argument('--date_ind', type=str, required=False)
        parser.add_argument('--date_fmt', type=str, required=False)


        if legacy_mode:
            parser.add_argument('--fig_fname_lab', type=str, required=False)
            parser.add_argument('--fig_fname_lab_2nd', type=str, required=False)
        parser.add_argument('--z_meth', type=str, help="z_slice, ss, nb, df, zm, or z_index for z level models")# Parse the argument

        parser.add_argument('--secdataset_proc', type=str, required=False)

        parser.add_argument('--hov_time', type=str, required=False)
        parser.add_argument('--do_cont', type=str, required=False)
        parser.add_argument('--do_grad', type=int, required=False)
        parser.add_argument('--trim_extra_files', type=int, required=False)
        parser.add_argument('--trim_files', type=int, required=False)

        parser.add_argument('--clim_sym', type=str, required=False)
        parser.add_argument('--clim_pair', type=str, required=False)
        parser.add_argument('--use_cmocean', type=str, required=False)

        parser.add_argument('--resample_freq', type=str, required=False)

        parser.add_argument('--ld_lst', type=str, required=False)
        parser.add_argument('--ld_lab_lst', type=str, required=False)
        parser.add_argument('--ld_nctvar', type=str, required=False)



        parser.add_argument('--fig_dir', type=str, required=False, help = 'if absent, will default to $PWD/tmpfigs')
        parser.add_argument('--fig_lab', type=str, required=False, help = 'if absent, will default to figs')
        parser.add_argument('--fig_cutout', type=str, required=False)
        #parser.add_argument('--fig_cutout', type=str, required=False)
        
        parser.add_argument('--justplot', type=str, required=False)
        parser.add_argument('--justplot_date_ind', type=str, required=False, help = 'comma separated values')
        parser.add_argument('--justplot_z_meth_zz', type=str, required=False, help = 'comma separated values, replace space with underscore - e.g. "Dataset_1"')
        parser.add_argument('--justplot_secdataset_proc', type=str, required=False, help = 'comma separated values')

        parser.add_argument('--verbose_debugging', type=str, required=False)
        parser.add_argument('--do_timer', type=str, required=False)
        parser.add_argument('--do_memory', type=str, required=False)
        parser.add_argument('--do_ensemble', type=str, required=False)
        parser.add_argument('--do_mask', type=str, required=False)
        parser.add_argument('--use_xarray_gdept', type=str, required=False)

        parser.add_argument('--Obs_dict', action='append', nargs='+')
        parser.add_argument('--Obs_hide', type=str, required=False)
        parser.add_argument('--files', action='append', nargs='+')
        parser.add_argument('--configs', action='append', nargs='+')
        parser.add_argument('--th', action='append', nargs='+')
        parser.add_argument('--figlabs', action='append', nargs='+')
        parser.add_argument('--forced_dim', action='append', nargs='+')
        parser.add_argument('--rename_var', action='append', nargs='+')



        args = parser.parse_args()# Print "Hello" + the user input argument


        
        # Handling of Bool variable types
        #

        
        if args.Obs_dict is None:
            Obs_dict_in = None
        else:
            Obs_dict_in = {}
            #for ss in np.unique([tmparr[0] for tmparr in args.Obs_dict])
            # cycle through input argument list elements
            for tmparr in args.Obs_dict:

                #take the dataset name
                tmp_datstr = 'Dataset ' + tmparr[0]
                #Add sub dictionary if not present 
                if tmp_datstr not in Obs_dict_in.keys():Obs_dict_in[tmp_datstr] = {}

                if len(tmparr)!=3:
                    print('arg error: (Obs_dict):', tmparr)

                tmp_datset = tmparr[1]
                tmp_files = tmparr[2]
                Obs_dict_in[tmp_datstr][tmp_datset] = tmp_files

            #pdb.set_trace()

        if args.preload_data is None:
            preload_data_in=True
        elif args.preload_data is not None:
            if args.preload_data.upper() in ['TRUE','T']:
                preload_data_in = bool(True)
            elif args.preload_data.upper() in ['FALSE','F']:
                preload_data_in = bool(False)
            else:                
                print('preload_data',args.preload_data)
                pdb.set_trace()

        if args.allow_diff_time is None:
            allow_diff_time_in=False
        elif args.allow_diff_time is not None:
            if args.allow_diff_time.upper() in ['TRUE','T']:
                allow_diff_time_in = bool(True)
            elif args.allow_diff_time.upper() in ['FALSE','F']:
                allow_diff_time_in = bool(False)
            else:                
                print('allow_diff_time',args.allow_diff_time)
                pdb.set_trace()

        if args.clim_sym is None:
            clim_sym_in=False
        elif args.clim_sym is not None:
            if args.clim_sym.upper() in ['TRUE','T']:
                clim_sym_in = bool(True)
            elif args.clim_sym.upper() in ['FALSE','F']:
                clim_sym_in = bool(False)
            else:                
                print('clim_sym',args.clim_sym)
                pdb.set_trace()

        if args.clim_pair is None:
            clim_pair_in=True
        elif args.clim_pair is not None:
            if args.clim_pair.upper() in ['TRUE','T']:
                clim_pair_in = bool(True)
            elif args.clim_pair.upper() in ['FALSE','F']:
                clim_pair_in = bool(False)
            else:                
                print('clim_pair',args.clim_pair)
                pdb.set_trace()

        if args.hov_time is None:
            hov_time_in=False
        elif args.hov_time is not None:
            if args.hov_time.upper() in ['TRUE','T']:
                hov_time_in = bool(True)
            elif args.hov_time.upper() in ['FALSE','F']:
                hov_time_in = bool(False)
            else:                
                print('hov_time',args.hov_time)
                pdb.set_trace()

        if args.fig_cutout is None:
            fig_cutout_in=True
        elif args.fig_cutout is not None:
            if args.fig_cutout.upper() in ['TRUE','T']:
                fig_cutout_in = bool(True)
            elif args.fig_cutout.upper() in ['FALSE','F']:
                fig_cutout_in = bool(False)
            else:                
                print('fig_cutout',args.fig_cutout)
                pdb.set_trace()

        if args.justplot is None:
            justplot_in=False
        elif args.justplot is not None:
            if args.justplot.upper() in ['TRUE','T']:
                justplot_in = bool(True)
            elif args.justplot.upper() in ['FALSE','F']:
                justplot_in = bool(False)
            else:                
                print('justplot',args.justplot)
                pdb.set_trace()

        if args.use_cmocean is None:
            use_cmocean_in=False
        elif args.use_cmocean is not None:
            if args.use_cmocean.upper() in ['TRUE','T']:
                use_cmocean_in = bool(True)
            elif args.use_cmocean.upper() in ['FALSE','F']:
                use_cmocean_in = bool(False)
            else:                
                print('use_cmocean',args.use_cmocean)
                pdb.set_trace()

        if args.verbose_debugging is None:
            verbose_debugging_in=False
        elif args.verbose_debugging is not None:
            if args.verbose_debugging.upper() in ['TRUE','T']:
                verbose_debugging_in = bool(True)
            elif args.verbose_debugging.upper() in ['FALSE','F']:
                verbose_debugging_in = bool(False)
            else:                
                print('verbose_debugging: ',args.verbose_debugging)
                pdb.set_trace()

        if args.do_timer is None:
            do_timer_in=False
        elif args.do_timer is not None:
            if args.do_timer.upper() in ['TRUE','T']:
                do_timer_in = bool(True)
            elif args.do_timer.upper() in ['FALSE','F']:
                do_timer_in = bool(False)
            else:                
                print('do_timer',args.do_timer)
                pdb.set_trace()
 

        if args.do_memory is None:
            do_memory_in=False
        elif args.do_memory is not None:
            if args.do_memory.upper() in ['TRUE','T']:
                do_memory_in = bool(True)
            elif args.do_memory.upper() in ['FALSE','F']:
                do_memory_in = bool(False)
            else:                
                print('do_timer',args.do_memory)
                pdb.set_trace()

        if args.do_ensemble is None:
            do_ensemble_in=False
        elif args.do_ensemble is not None:
            if args.do_ensemble.upper() in ['TRUE','T']:
                do_ensemble_in = bool(True)
            elif args.do_ensemble.upper() in ['FALSE','F']:
                do_ensemble_in = bool(False)
            else:                
                print('do_timer',args.do_timer)
                pdb.set_trace()
 
 
        if args.do_mask is None:
            do_mask_in=False
        elif args.do_mask is not None:
            if args.do_mask.upper() in ['TRUE','T']:
                do_mask_in = bool(True)
            elif args.do_mask.upper() in ['FALSE','F']:
                do_mask_in = bool(False)
            else:                
                print('do_mask',args.do_mask)
                pdb.set_trace()
        if args.do_grad is None:
            do_grad_in=0
        else:
            do_grad_in=args.do_grad

        if args.do_cont is None:
            do_cont_in=False
        elif args.do_cont is not None:
            if args.do_cont.upper() in ['TRUE','T']:
                do_cont_in = bool(True)
            elif args.do_cont.upper() in ['FALSE','F']:
                do_cont_in = bool(False)
            else:                
                print('do_cont',args.do_cont)
                pdb.set_trace()



        if args.trim_extra_files is None:
            trim_extra_files=False
        elif args.trim_extra_files is not None:
            if args.trim_extra_files.upper() in ['TRUE','T']:
                trim_extra_files = bool(True)
            elif args.trim_extra_files.upper() in ['FALSE','F']:
                trim_extra_files = bool(False)
            else:                
                print('trim_extra_files',args.trim_extra_files)
                pdb.set_trace()
 


        if args.trim_files is None:
            trim_files=True
        elif args.trim_files is not None:
            if args.trim_files.upper() in ['TRUE','T']:
                trim_files = bool(True)
            elif args.trim_files.upper() in ['FALSE','F']:
                trim_files = bool(False)
            else:                
                print('trim_files',args.trim_files)
                pdb.set_trace()
 



        if args.Obs_hide is None:
            Obs_hide_in=False
        elif args.Obs_hide is not None:
            if args.Obs_hide.upper() in ['TRUE','T']:
                Obs_hide_in = bool(True)
            elif args.Obs_hide.upper() in ['FALSE','F']:
                Obs_hide_in = bool(False)
            else:                
                print('Obs_hide',args.Obs_hide)
                pdb.set_trace()

        if args.use_xarray_gdept is None:
            use_xarray_gdept_in=False
        elif args.use_xarray_gdept is not None:
            if args.use_xarray_gdept.upper() in ['TRUE','T']:
                use_xarray_gdept_in = bool(True)
            elif args.use_xarray_gdept.upper() in ['FALSE','F']:
                use_xarray_gdept_in = bool(False)
            else:                
                print('use_xarray_gdept',args.use_xarray_gdept)
                pdb.set_trace()


        '''

        if args.boolvar is None:
            boolvar_in=True
        elif args.boolvar is not None:
            if args.boolvar.upper() in ['TRUE','T']:
                boolvar_in = bool(True)
            elif args.boolvar.upper() in ['FALSE','F']:
                boolvar_in = bool(False)
            else:                
                print(args.boolvar)
                pdb.set_trace()





        config = None
        fname_lst = None
        fname_lst_2nd = None
        config_2nd = None
        U_fname_lst = None
        V_fname_lst = None
        U_fname_lst_2nd = None
        U_fname_lst_2nd = None
        thin = 1
        thin_2nd = 1
        thin_x0=0
        thin_x1=None
        thin_y0=0
        thin_y1=None
        thin_files = 1
        thin_files_0 = 0
        thin_files_1 = None



        '''
        #set default values for None


        #pdb.set_trace()


        #if (args.fig_dir) is None: args.fig_dir=script_dir + '/tmpfigs'
        if (args.fig_dir) is None: args.fig_dir='./tmpfigs'
        if (args.fig_lab) is None: args.fig_lab='figs'
        if (args.ld_nctvar) is None: args.ld_nctvar='time_counter'



        if (args.date_fmt) is None: args.date_fmt='%Y%m%d'

        #print('justplot',args.justplot)
        
        #if args.thin is None: args.thin=1
        #if args.thin_2nd is None: args.thin_2nd=1
        #if args.thin_files is None: args.thin_files=1
        #if args.thin_files_0 is None: args.thin_files_0=0
        #if args.thin_files_1 is None: args.thin_files_1=None

        #if args.thin_x0 is None: args.thin_files_0=0
        #if args.thin_x1 is None: args.thin_files_1=None
        #if args.thin_y0 is None: args.thin_files_0=0
        #if args.thin_y1 is None: args.thin_files_1=None


        print(args.fname_lst)
        fname_lst = glob.glob(args.fname_lst)
        fname_lst.sort()

        if legacy_mode:
            #Deal with file lists
            fname_lst_2nd = None
            U_fname_lst = None
            V_fname_lst = None
            U_fname_lst_2nd = None
            V_fname_lst_2nd = None
            WW3_fname_lst = None
            WW3_fname_lst_2nd = None

            load_second_files = False
            
            if (args.fname_lst_2nd) is not None:
                fname_lst_2nd = glob.glob(args.fname_lst_2nd)
                load_second_files = True
            if (args.U_fname_lst) is not None:U_fname_lst = glob.glob(args.U_fname_lst)
            if (args.V_fname_lst) is not None:V_fname_lst = glob.glob(args.V_fname_lst)
            if (args.U_fname_lst_2nd) is not None:
                U_fname_lst_2nd = glob.glob(args.U_fname_lst_2nd)
                load_second_files = True
            if (args.V_fname_lst_2nd) is not None:
                V_fname_lst_2nd = glob.glob(args.V_fname_lst_2nd)
                load_second_files = True
            if (args.WW3_fname_lst) is not None:WW3_fname_lst = glob.glob(args.WW3_fname_lst)
            if (args.WW3_fname_lst_2nd) is not None:
                WW3_fname_lst_2nd = glob.glob(args.WW3_fname_lst_2nd)
                load_second_files = True

            if (fname_lst_2nd) is not None:fname_lst_2nd.sort()
            if (U_fname_lst) is not None:U_fname_lst.sort()
            if (V_fname_lst) is not None:V_fname_lst.sort()
            if (U_fname_lst_2nd) is not None:U_fname_lst_2nd.sort()
            if (V_fname_lst_2nd) is not None:V_fname_lst_2nd.sort()

            if (WW3_fname_lst) is not None:WW3_fname_lst.sort()
            if (WW3_fname_lst_2nd) is not None:WW3_fname_lst_2nd.sort()


            if len(fname_lst) == 0: 
                print('')
                print('no files passed')
                print('')
                print('')
                print('=======================================================')
                pdb.set_trace()


        if legacy_mode:
            #load_second_files = False

            configd = {}
            configd[1] = args.config
            if load_second_files:
                if (args.config_2nd) is None: 
                    configd[2] = configd[1]
                else:
                    configd[2] =args.config_2nd
            '''
            if args.config_2nd is not None: 
                configd[2]

            configd[2] = None
            if 'config_2nd' in args:
                if args.config_2nd is not None: 
                    configd[2] = args.config_2nd
            
                    load_second_files = True
            
            
            if 'fname_lst_2nd' in args:
                if 'config_2nd' in args:
                    if args.config_2nd is None: 
                        configd[2] = None
                
                load_second_files = True
            '''
            #if 2 in configd.keys():
        
            fname_dict = {}
            fname_dict['Dataset 1'] = {}
            fname_dict['Dataset 1']['T'] = fname_lst
            if U_fname_lst is not None: fname_dict['Dataset 1']['U'] = U_fname_lst
            if V_fname_lst is not None: fname_dict['Dataset 1']['V'] = V_fname_lst
            if WW3_fname_lst is not None: fname_dict['Dataset 1']['WW3'] = WW3_fname_lst

            #pdb.set_trace()
            if load_second_files: 
                fname_dict['Dataset 2'] = {}
                fname_dict['Dataset 2']['T'] = fname_lst_2nd
                if U_fname_lst is not None: fname_dict['Dataset 2']['U'] = U_fname_lst_2nd
                if V_fname_lst is not None: fname_dict['Dataset 2']['V'] = V_fname_lst_2nd
                if WW3_fname_lst is not None: fname_dict['Dataset 2']['WW3'] = WW3_fname_lst_2nd







        load_second_files = False
        fname_dict = {}
        fname_dict['Dataset 1'] = {}
        fname_dict['Dataset 1']['T'] = fname_lst
        
        if args.files is not None:
            #for ss in np.unique([tmparr[0] for tmparr in args.Obs_dict])
            # cycle through input argument list elements
            for tmparr in args.files:

                #take the dataset name
                tmp_datstr = 'Dataset ' + tmparr[0]
                #Add sub dictionary if not present 
                if tmp_datstr not in fname_dict.keys():fname_dict[tmp_datstr] = {}
                if len(tmparr)!=3:
                    print('arg error: (files):', tmparr)

                tmp_grid = tmparr[1]
                tmp_files = tmparr[2]
                fname_dict[tmp_datstr][tmp_grid] = np.sort(glob.glob(tmp_files))
                if len(fname_dict[tmp_datstr][tmp_grid]) == 0: 
                    print('\n\nNo Files for %s %s = %s\n\n'%(tmp_datstr,tmp_grid,tmp_files))
                    pdb.set_trace()
            if len(fname_dict.keys()) > 1:load_second_files = True


        dataset_lst = [ ss for ss in fname_dict.keys() ] 
        nDataset = len(dataset_lst)

        configd = {}
        fig_lab_d = {}
            #for tmp_datstr in dataset_lst:
        for ii, tmp_datstr in enumerate(dataset_lst):
            configd[ii+1] = args.config
            fig_lab_d['Dataset %i'%(ii+1) ] = None

        if args.configs is not None:
            
            for tmparr in args.configs:
                if len(tmparr)!=2:
                    print('arg error: (configs):', tmparr)
                #pdb.set_trace()
                configd[int(tmparr[0])] = tmparr[1]

        configlst = np.array([configd[ss] for ss in (configd)])
        uniqconfig = np.unique(configlst)

    

        if legacy_mode:
            fig_lab_d = {}
            #for tmp_datstr in dataset_lst:
            fig_lab_d['Dataset 1'] = None
            #pdb.set_trace()

            if 'fig_fname_lab' in args: fig_lab_d['Dataset 1'] = args.fig_fname_lab
            if load_second_files: 
                if 'fig_fname_lab_2nd' in args:fig_lab_d['Dataset 2'] = args.fig_fname_lab_2nd
                
            #if fig_fname_lab is not None: fig_lab_d['Dataset 1'] = fig_fname_lab
            #if fig_fname_lab_2nd is not None: fig_lab_d['Dataset 2'] = fig_fname_lab_2nd
            #del(fig_fname_lab)
            #del(fig_fname_lab_2nd)

            #pdb.set_trace()


        if args.figlabs is not None:
            fig_lab_d = {}
            fig_lab_d['Dataset 1'] = None

            for tmparr in args.figlabs:
                if len(tmparr)!=2:
                    print('arg error: (figlabs):', tmparr)
                tmp_datstr = 'Dataset ' + tmparr[0]
                #pdb.set_trace()
                fig_lab_d[tmp_datstr] = tmparr[1]


        if legacy_mode:

            thd = {}
            thd[1] = {}
            thd[1]['df'] = 1
            thd[1]['f0'] = 0
            thd[1]['f1'] = None
            if (args.thin_files)   is not None: thd[1]['df'] = args.thin_files
            if (args.thin_files_0) is not None: thd[1]['f0'] = args.thin_files_0
            if (args.thin_files_1) is not None: thd[1]['f1'] = args.thin_files_1

            thd[1]['dx'] = 1
            thd[1]['dy'] = 1
            thd[1]['x0'] = 0
            thd[1]['x1'] = None
            thd[1]['y0'] = 0
            thd[1]['y1'] = None
    
            if (args.thin)    is not None: thd[1]['dx'] = args.thin
            if (args.thin)    is not None: thd[1]['dy'] = args.thin
            if (args.thin_x0) is not None: thd[1]['x0'] = args.thin_x0
            if (args.thin_x1) is not None: thd[1]['x1'] = args.thin_x1
            if (args.thin_y0) is not None: thd[1]['y0'] = args.thin_y0
            if (args.thin_y1) is not None: thd[1]['y1'] = args.thin_y1

            if load_second_files:
                thd[2] = {}
                thd[2]['df'] = thd[1]['df']
                thd[2]['f0'] = thd[1]['f0']
                thd[2]['f1'] = thd[1]['f1']
                thd[2]['dx'] = thd[1]['dx']
                thd[2]['dy'] = thd[1]['dy']
                thd[2]['x0'] = thd[1]['x0']
                thd[2]['x1'] = thd[1]['x1']
                thd[2]['y0'] = thd[1]['y0']
                thd[2]['y1'] = thd[1]['y1']



                if (args.thin_2nd)    is not None: thd[2]['dx'] = args.thin_2nd
                if (args.thin_2nd)    is not None: thd[2]['dy'] = args.thin_2nd
                if (args.thin_x0_2nd) is not None: thd[2]['x0'] = args.thin_x0_2nd
                if (args.thin_x1_2nd) is not None: thd[2]['x1'] = args.thin_x1_2nd
                if (args.thin_y0_2nd) is not None: thd[2]['y0'] = args.thin_y0_2nd
                if (args.thin_y1_2nd) is not None: thd[2]['y1'] = args.thin_y1_2nd


            #pdb.set_trace()
        
        # set up empty th dictionary
        thd = {}
        thd[1] = {}
        thd[1]['df'] = 1
        thd[1]['f0'] = 0
        thd[1]['f1'] = None

        thd[1]['dx'] = 1
        thd[1]['dy'] = 1
        thd[1]['x0'] = 0
        thd[1]['x1'] = None
        thd[1]['y0'] = 0
        thd[1]['y1'] = None

        thd[1]['pxy'] = None

            

        #pdb.set_trace()

        # Copy dummy values for all datasets
        for ii in range(len(dataset_lst)-1):
            thd[ii+2] = thd[1].copy()

        # update with arguements.
        # if argument entered for one dataset, copy to all subsequent datasets with the same config
        if args.th is not None:
            th_args = (args.th)
            th_args.sort()

            print('th arguments:',th_args)
            #for tmparr in args.th:
            for tmparr in th_args:
                #pdb.set_trace()
                if len(tmparr)!=3:
                    print('arg error: (th):', tmparr)
                #take the dataset name
                '''
                tmp_datstr = int(tmparr[0])
                tmp_thvar = tmparr[1]
                tmp_thval = tmparr[2]
                if tmp_thval.lower() == 'none':
                    tmp_thval = None
                else:
                    tmp_thval = int(tmp_thval)
                #pdb.set_trace()
                if tmp_thvar in ['dxy','dx','dy']:
                    thd[tmp_datstr]['dx'] = tmp_thval
                    thd[tmp_datstr]['dy'] = tmp_thval
                else:
                    thd[tmp_datstr][tmp_thvar] = tmp_thval
                '''
                tmp_datstr = int(tmparr[0])
                tmpconfig = configd[tmp_datstr]
                tmp_thvar = tmparr[1]
                tmp_thval = tmparr[2]
                if tmp_thval.lower() == 'none':
                    tmp_thval = None
                else:
                    tmp_thval = int(tmp_thval)
                for dsi in range(tmp_datstr,nDataset+1):
                    if configd[dsi] == tmpconfig:
                        #pdb.set_trace()
                        if tmp_thvar in ['dxy','dx','dy']:
                            thd[dsi]['dx'] = tmp_thval
                            thd[dsi]['dy'] = tmp_thval
                        else:
                            thd[dsi][tmp_thvar] = tmp_thval
        
        # of orca model cut off upper values as non incrementing longitude
        for cfi in configd.keys():
            if configd[cfi] is None: continue
            if configd[cfi].upper() in ['ORCA025','ORCA025EXT','ORCA025ICE']: 
                if thd[cfi]['y1'] is None: thd[cfi]['y1'] = -2
            if configd[cfi].upper() in ['ORCA12','ORCA12ICE']: 
                if thd[cfi]['y1'] is None: thd[cfi]['y1'] = -200

        #pdb.set_trace()
        # print out thd
        #for dsi in range(1,nDataset+1): print(thd[dsi])

        # ensure all configs have same thinning, use lowest dataset value
        if nDataset > 1:
            for tmpconfig in uniqconfig:
                tmpconfigind = np.where(configlst == tmpconfig)[0] + 1
                for dsi in tmpconfigind[1:]:
                    thd[dsi] = thd[tmpconfigind[0]]
            
        # print out thd
        for dsi in range(1,nDataset+1): print(thd[dsi])


        #configlst = np.array([configd[ss] for ss in (configd)])
        #uniqconfig = np.unique(configlst)


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


        ## xarr_rename_master_dict = {'votemper':'temperature','vosaline':'salinity'}

        xarr_rename_master_dict_in = None
        if args.rename_var is not None:
            xarr_rename_master_dict_in = {}
            for tmpvarrenlst in args.rename_var:
                
                if len(tmpvarrenlst)<2:
                    print('arg error: (rename_var):', tmpvarrenlst) 
                for tvr in tmpvarrenlst[1:]: xarr_rename_master_dict_in[tvr] = tmpvarrenlst[0]

        

        # If files have more than grid, with differing dimension for each, you can enforce the dimenson for each grid.
        # For example, the SMHI BAL-MFC NRT system (BALMFCorig) hourly surface files hvae the T, U, V and T_inner grid in the same file. 
        # Load the smae file in for each grid:
        # Nslvdev BALMFCorig NS01_SURF_2025020912_1-24H.nc 
        # --files 1 U NS01_SURF_2025020912_1-24H.nc --files 1 V NS01_SURF_2025020912_1-24H.nc --files 1 Ti NS01_SURF_2025020912_1-24H.nc 
        # .....   --th 1 dxy 
        # and enforce which dimensions are used for the T, U, V and T_inner grid (x,y,z and T)
        #--forced_dim U x x_grid_U y y_grid_U --forced_dim V x x_grid_V y y_grid_V --forced_dim T x x_grid_T y y_grid_T --forced_dim Ti x x_grid_T_inner y y_grid_T_inner
        #
        # force_dim_d_in = 
        # {'U': {'x': 'x_grid_U', 'y': 'y_grid_U'}, 'V': {'x': 'x_grid_V', 'y': 'y_grid_V'}, 'T': {'x': 'x_grid_T', 'y': 'y_grid_T'}, 'x': {'x_grid_T_inner': 'y'}}
        #
        # If you are comparing different domains, and this only needs to be added to one dataset, add the dataset number before the grid:
        #
        # --forced_dim 1 U x x_grid_U y y_grid_U --forced_dim 1 V x x_grid_V y y_grid_V --forced_dim 1 T x x_grid_T y y_grid_T --forced_dim 1 Ti x x_grid_T_inner y y_grid_T_inner
        # force_dim_d_in =
        #   {1: {'1': {'Ti': 'x', 'x_grid_T_inner': 'y'}}, 2: {'1': {'Ti': 'x', 'x_grid_T_inner': 'y'}}}


        '''
        force_dim_d_in = None
        if args.forced_dim is not None:
            force_dim_d_in = {}
            force_dim_d['all_grid'] = True
            for tmpfdimlst in args.forced_dim:
                
                if (len(tmpfdimlst)%2 != 1) | len(tmpfdimlst)<2:
                    print('arg error: (forced_dim):', tmpvarrenlst) 

                tmpdimgrid = tmpfdimlst[0]
                force_dim_d_in[tmpdimgrid] = {}
                nfdimlst=len(tmpfdimlst[1:])/2
                for tfdi in range(int(nfdimlst)): force_dim_d_in[tmpdimgrid][tmpfdimlst[(tfdi*2)+1]] = tmpfdimlst[(tfdi*2)+2]
                
        '''

        #for ii, tmp_datstr in enumerate(dataset_lst):
        #pdb.set_trace() 
        force_dim_d_in = None
        if args.forced_dim is not None:
            force_dim_d_in = {}
            for dsi in np.arange(1,nDataset+1): 
                force_dim_d_in[dsi] = {}
            #force_dim_d['all_grid'] = True
            for tmpfdimlst in args.forced_dim:
                
                #if (len(tmpfdimlst)%2 != 1) | len(tmpfdimlst)<2:
                #    print('arg error: (forced_dim):', tmpvarrenlst) 


                if len(tmpfdimlst)<2:
                    print('arg error: (forced_dim):', tmpvarrenlst) 
                
                
                #if the first entry is not a dataset number
                if tmpfdimlst[0] not in np.arange(nDataset):#
                    tmpdimgrid = tmpfdimlst[0]
                    tmp_dsi_lst = np.arange(1,nDataset+1)
                # if the first entry is a datatset number:
                else:
                    tmpdimgrid = tmpfdimlst[1]
                    tmp_dsi_lst = [tmpfdimlst[0]]
                    tmpdimgrid = tmpdimgrid[1:]
                
                for dsi in tmp_dsi_lst: 
                    force_dim_d_in[dsi][tmpdimgrid] = {}
                    nfdimlst=len(tmpfdimlst[1:])/2
                    for tfdi in range(int(nfdimlst)): force_dim_d_in[dsi][tmpdimgrid][tmpfdimlst[(tfdi*2)+1]] = tmpfdimlst[(tfdi*2)+2]
                print(tmpfdimlst, force_dim_d_in)    

        #pdb.set_trace()
        nemo_slice_zlev(zlim_max = args.zlim_max,
            fig_lab_d = fig_lab_d,configd = configd,thd = thd,fname_dict = fname_dict,load_second_files = load_second_files,
            clim_sym = clim_sym_in, clim = args.clim, clim_pair = clim_pair_in,hov_time = hov_time_in,
            allow_diff_time = allow_diff_time_in,preload_data = preload_data_in,
            do_grad = do_grad_in,do_cont = do_cont_in,trim_extra_files = trim_extra_files,trim_files = trim_files,
            use_cmocean = use_cmocean_in, date_fmt = args.date_fmt,
            justplot = justplot_in,justplot_date_ind = args.justplot_date_ind,
            justplot_secdataset_proc = args.justplot_secdataset_proc,
            justplot_z_meth_zz = args.justplot_z_meth_zz,
            ii = args.ii, jj = args.jj, ti = args.ti, zz = args.zz, 
            lon_in = args.lon, lat_in = args.lat, date_in_ind = args.date_ind,
            var = args.var, z_meth = args.z_meth,
            xlim = args.xlim,ylim = args.ylim,
            secdataset_proc = args.secdataset_proc,
            ld_lst = args.ld_lst, ld_lab_lst = args.ld_lab_lst, ld_nctvar = args.ld_nctvar,
            resample_freq = args.resample_freq,
            Obs_dict = Obs_dict_in,Obs_hide = Obs_hide_in,
            fig_dir = args.fig_dir, fig_lab = args.fig_lab,fig_cutout = fig_cutout_in,
            verbose_debugging = verbose_debugging_in,do_timer = do_timer_in,do_memory = do_memory_in,do_ensemble = do_ensemble_in,do_mask = do_mask_in,
            use_xarray_gdept = use_xarray_gdept_in,
            force_dim_d = force_dim_d_in,xarr_rename_master_dict=xarr_rename_master_dict_in)



        exit()

    
if __name__ == "__main__":
    main()

