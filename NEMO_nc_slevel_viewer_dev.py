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

from NEMO_nc_slevel_viewer_lib import add_default_nice_varname, process_argparse_define_time

from NEMO_nc_slevel_viewer_lib import extract_time_from_xarr,resample_xarray

# Data loading modules
#from NEMO_nc_slevel_viewer_lib import reload_data_instances
from NEMO_nc_slevel_viewer_lib import dataset_comp_func
from NEMO_nc_slevel_viewer_lib import reload_data_instances_time,reload_hov_data_comb_time,reload_ts_data_comb_time
from NEMO_nc_slevel_viewer_lib import reload_map_data_comb,reload_ew_data_comb,reload_ns_data_comb,reload_pf_data_comb
#from NEMO_nc_slevel_viewer_lib import reload_hov_data_comb,reload_ts_data_comb
from NEMO_nc_slevel_viewer_lib import reload_time_dist_data_comb_time


# Data processing modules
from NEMO_nc_slevel_viewer_lib import grad_horiz_ns_data,grad_horiz_ew_data
from NEMO_nc_slevel_viewer_lib import grad_vert_ns_data,grad_vert_ew_data,grad_vert_hov_prof_data

from NEMO_nc_slevel_viewer_lib import field_gradient_2d, vector_div, vector_curl,sw_dens

# Data manipulation modules
#from NEMO_nc_slevel_viewer_lib import rotated_grid_from_amm15, reduce_rotamm15_grid,regrid_2nd_thin_params,regrid_iijj_ew_ns
from NEMO_nc_slevel_viewer_lib import interp1dmat_create_weight

# Plotting modules
from NEMO_nc_slevel_viewer_lib import get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region,get_colorbar_values
from NEMO_nc_slevel_viewer_lib import scale_color_map,lon_lat_to_str,current_barb,get_pnts_pcolor_in_region
from NEMO_nc_slevel_viewer_lib import profile_line

from NEMO_nc_slevel_viewer_lib import pop_up_opt_window,pop_up_info_window,get_help_text,jjii_from_lon_lat

from NEMO_nc_slevel_viewer_lib import calc_ens_stat_2d, calc_ens_stat_3d,calc_ens_stat_map

from NEMO_nc_slevel_viewer_lib import int_ind_wgt_from_xypos, ind_from_lon_lat


from NEMO_nc_slevel_viewer_lib import load_nemo_slice_zlev_helptext, load_NEMO_nc_viewer_parser
from NEMO_nc_slevel_viewer_lib import process_argparse_bool 
from NEMO_nc_slevel_viewer_lib import process_argparse_fname_dict,process_argparse_configd
from NEMO_nc_slevel_viewer_lib import process_argparse_dataset_lab_d,process_argparse_thd
from NEMO_nc_slevel_viewer_lib import process_argparse_Obs_dict,process_argparse_rename_var
from NEMO_nc_slevel_viewer_lib import process_argparse_forced_dim,process_argparse_EOS,process_argparse_Obs_type_hide


from NEMO_nc_slevel_viewer_lib import obs_reset_sel # load_ops_prof_TS, load_ops_2D_xarray
from NEMO_nc_slevel_viewer_lib import Obs_load_init_files_dict,Obs_set_empty_dict,Obs_setup_Obs_vis_d,Obs_setup_Obs_JULD_datetime_dict
from NEMO_nc_slevel_viewer_lib import Obs_reload_obs,obs_load_selected_point, obs_get_Obs_Type_load_lst

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
    dataset_lab_d = None,configd = None,thd = None,fname_dict = None,load_second_files = False,
    xlim = None, ylim = None, tlim = None, clim = None,
    ii = None, jj = None, ti = None, zz = None, zi = None, 
    lon_in = None, lat_in = None, date_in_ind = None, date_fmt = '%Y%m%d',
    Time_Diff = None,
    cutxind = None, cutyind=None,
    z_meth = None,
    secdataset_proc = 'Dataset 1',
    hov_time = False, do_cont = False, do_grad = 0,
    allow_diff_time = False,define_time_dict = None, 
    preload_data = True,
    ld_lst = None, ld_nctvar = 'time_counter',ld_lab_lst = '-36,-12,012,036,060,084,108,132',
    clim_sym = None, clim_pair = True,use_cmocean = False,
    fig_dir = None,fig_lab = 'figs',fig_cutout = True, 
    justplot = False, justplot_date_ind = None,justplot_z_meth_zz = None,justplot_secdataset_proc = None,
    fig_fname_lab = None, 
    trim_files = True,
    trim_extra_files = False,
    vis_curr = -1, vis_curr_meth = 'barb',
    resample_freq = None,
    verbose_debugging = False,do_timer = True,do_memory = True,do_ensemble = False,
    Obs_dict = None, Obs_reloadmeth = 2,Obs_hide = False,
    Obs_hide_edges = None, Obs_pair_loc = None, Obs_AbsAnom = None,
    Obs_anom_clim = None,Obs_Type_load_dict = None,Obs_show_with_diff_var = None, 
    do_MLD = True,do_mask = False,
    use_xarray_gdept = True,
    force_dim_d = None,xarr_rename_master_dict=None,
    EOS_d = None,gr_1st = None,do_match_time = True,
    do_addtimedim = None, do_all_WW3 = False):

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

    '''

    cutout_data = False
    if cutxind is None:
        cutxind = [0,None]
    else:
        cutout_data = True
    if cutyind is None:
        cutyind = [0,None]
    else:
        cutout_data = True
    
    pdb.set_trace()

    cutxind = thd[1]['cutx0'],thd[1]['cutx1']
    cutyind = thd[1]['cuty0'],thd[1]['cuty1']
    
    # global model (e.g. orca12) enforce th y1 to be e.g. -200, to avoid polar convergence
    if cutyind[1] is not None: thd[1]['y1'] = None
    
    if cutxind[0]!=0:cutout_data = True
    if cutyind[0]!=0:cutout_data = True
    if cutxind[1] is not None:cutout_data = True
    if cutyind[1] is not None:cutout_data = True
    '''

    if EOS_d is None:
        EOS_d = {}
        EOS_d['do_TEOS_EOS_conv'] = False




    ens_stat_lst = ['EnsMean','EnsStd','EnsVar','EnsCnt']
    if do_ensemble:
        Ens_stat = None

    # if arguement Obs_dict is not None, set do_Obs to true
    
    if Obs_dict is None:
        do_Obs = False
    else:
        do_Obs = True
        #pdb.set_trace()
        
        # check if Obs_dict values are strings (for file names), or a dictionaries (i.e. obs loaded externally)
        for tmp_datstr in Obs_dict.keys():
            for ob_var in Obs_dict[tmp_datstr].keys():
                # True if string (i.e. path to file name)
                Obs_dict_is_str = isinstance(Obs_dict[tmp_datstr][ob_var] , str)
        
        #Obs_dict_is_str defaults to 2. 
        if Obs_dict_is_str == False:
            Obs_reloadmeth = -1
        elif Obs_dict_is_str:
            # if Obs_dict_is_str == True (path to obs files), keep Obs_dict_is_str=2, and chnage copy Obs_dict to Obs_fname
            Obs_fname = Obs_dict.copy()

            

            for tmp_datstr in Obs_fname.keys():
                for ob_var in Obs_fname[tmp_datstr].keys():
                    Obs_fname[tmp_datstr][ob_var] = np.sort(glob.glob(Obs_fname[tmp_datstr][ob_var]))
                    if len(Obs_fname[tmp_datstr][ob_var]) == 0: print('\n\nNo Obs files found in %s %s: %s\n\n'%(tmp_datstr,ob_var,  Obs_fname[tmp_datstr][ob_var]))
           




            #Obs_dict = {}

            
            if Obs_reloadmeth == 0:
                Obs_dict = Obs_load_init_files_dict(Obs_fname,Obs_Type_load_dict)
                '''       
                Obs_dict = {}
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

                        Obs_dict[tmp_datstr][ob_var]['JULD'] = np.array(Obs_dict[tmp_datstr][ob_var]['JULD'])
            
                '''
            elif Obs_reloadmeth > 0:

                Obs_dict = Obs_set_empty_dict(Obs_fname)
                '''
                Obs_dict = {}
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
        

                '''
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

    tmpdataset_oper_lst =  ['-','/','%']

    # create colours and line styles for plots
    Dataset_col,Dataset_col_diff,linestyle_str = create_col_lst(nDataset,tmpdataset_oper_lst)


    for tmp_datstr in Dataset_lst:
        if tmp_datstr in dataset_lab_d.keys():
            if dataset_lab_d[tmp_datstr] is None: dataset_lab_d[tmp_datstr] = tmp_datstr
        else:
            dataset_lab_d[tmp_datstr] = tmp_datstr


    axis_scale = 'Auto'
    grad_horiz_vert_wgt = False
    if do_grad is None: do_grad = 0
    if do_cont is None: do_cont = True
    


    grad_meth=0
    grad_2d_meth = 0
    grad_abs_pre = False
    grad_abs_post = False
    grad_regrid_xy = False
    grad_dx_d_dx = False




    if verbose_debugging:
        print('======================================================')
        print('======================================================')
        print('=== Debugging printouts: verbose_debugging = True  ===')
        print('======================================================')
        print('======================================================')

    
    #Default variable for U and V flist
    tmp_var_U, tmp_var_V = 'vozocrtx','vomecrty'


    #z_meth_mat = ['z_slice','ss','nb','df','zm']

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

    #pdb.set_trace()
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
                """
                if tmp_datstr1>tmp_datstr2:                    
                    tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)
                    secdataset_proc_list.append(tmp_diff_str_name)
                    Dataset_col_diff_dict[tmp_diff_str_name] = Dataset_col_diff[cnt_diff_str_name]
                    cnt_diff_str_name = cnt_diff_str_name+1
                """
                if tmp_datstr1!=tmp_datstr2:
                #if ((nDataset<6)&(th_d_ind1 != th_d_ind2))|((nDataset>=6)&(th_d_ind1 > th_d_ind2)):
                    '''
                    tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2)
                    secdataset_proc_list.append(tmp_diff_str_name)
                    Dataset_col_diff_dict[tmp_diff_str_name] = Dataset_col_diff[cnt_diff_str_name]
                    cnt_diff_str_name = cnt_diff_str_name+1
                    '''
                    for tmpdataset_oper in tmpdataset_oper_lst:
                        tmp_diff_str_name = 'Dat%i%sDat%i'%(th_d_ind1,tmpdataset_oper,th_d_ind2)
                        if tmpdataset_oper == '-':
                            if tmp_datstr1>tmp_datstr2:
                                secdataset_proc_list.append(tmp_diff_str_name)
                        #secdataset_proc_list.append(tmp_diff_str_name)
                        # color 
                        Dataset_col_diff_dict[tmp_diff_str_name] = Dataset_col_diff[cnt_diff_str_name]
                        cnt_diff_str_name = cnt_diff_str_name+1

    if secdataset_proc is None: secdataset_proc = Dataset_lst[0]

    if load_second_files == False:
        clim_pair = False
    
    if justplot is None: justplot = False

    if hov_time is None: hov_time = True

    print('thin: %i; thin_files: %i; hov_time: %s; '%(thd[1]['dx'],thd[1]['df'],hov_time))
    pxy = thd[1]['pxy']
    print('maximum pixels plotted in x and y directions pxy:',pxy)

    nlon_amm7 = 297
    nlat_amm7 = 375
    nlon_amm15 = 1458
    nlat_amm15 = 1345





    # create a dictionary with all the config info in it
    config_fnames_dict = create_config_fnames_dict(configd,Dataset_lst,script_dir)

    init_timer.append((datetime.now(),'Indices set'))



    # check if there are any LBCs
    do_LBC_d = {}
    LBC_coord_d = {}
    do_LBC = False

    
    #for tmp_datstr in Dataset_lst:
    #    th_d_ind = int(tmp_datstr[8:])
    #    pdb.set_trace()
    
    for cfi in configd.keys():
        if configd[cfi][-3:].upper() == 'LBC':
            do_LBC_d[cfi] = True
            do_LBC = True

        else:
            do_LBC_d[cfi] = False

    if do_LBC == True:
        LBC_coord_d = {}
        for cfi in configd.keys():
            if do_LBC_d[cfi]:
                LBC_coord_d[cfi] = {}
                for LBC_set in range(1,11):
                    LBC_set_name = 'bdy_coord_%i_file'%LBC_set
                    if LBC_set_name in config_fnames_dict[configd[cfi]].keys():
                        LBC_coord_d[cfi][LBC_set] = {}
                            
                        rootgrp = Dataset(config_fnames_dict[configd[cfi]][LBC_set_name], 'r')
                        for LBC_var in rootgrp.variables.keys(): LBC_coord_d[cfi][LBC_set][LBC_var] = rootgrp.variables[LBC_var][:]                        
                        rootgrp.close()
    #pdb.set_trace()


    
    if gr_1st is None:
        if do_LBC == True:
            if do_LBC_d[1]:
                gr_1st = 'T_1'
        else:
            gr_1st = 'T'
    

    #pdb.set_trace()

    init_timer.append((datetime.now(),'LBC prep work'))

    
    z_meth_default = config_fnames_dict[configd[1]]['z_meth_default']
    
    #set ncvariable names for mesh files
    ncgdept,nce1t,nce2t,nce3t,ncglamt,ncgphit = create_gdept_ncvarnames(config_fnames_dict,configd)
   
    init_timer.append((datetime.now(),'Config files read'))

    # create dictionary with mesh files handles
    rootgrp_gdept_dict = create_rootgrp_gdept_dict(config_fnames_dict,Dataset_lst,configd, use_xarray_gdept = use_xarray_gdept)
    
    init_timer.append((datetime.now(),'Gdept opened'))

    #pdb.set_trace()


    if z_meth is None:
        z_meth = z_meth_default
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
    var_d,var_dim,var_grid,ncvar_d,ncdim_d,time_d  = connect_to_files_with_xarray(Dataset_lst,fname_dict,xarr_dict,nldi,ldi_ind_mat, 
                                                                                  ld_lab_mat,ld_nctvar,force_dim_d = force_dim_d,
                                                                                  xarr_rename_master_dict=xarr_rename_master_dict,
                                                                                  gr_1st = gr_1st,do_addtimedim = do_addtimedim,
                                                                                  do_all_WW3 = do_all_WW3,
                                                                                  define_time_dict = define_time_dict)
    
    #pdb.set_trace()
    # tmp = xarr_dict['Dataset 1']['T'][0].groupby('time_counter.year').groupby('time_counter.month').mean('time_counter') 
    
    #nctime = xarr_dict['Dataset 1']['T'][0].variables['time_counter']
    #xarr_dict['Dataset 1']['T'][0] = xarr_dict['Dataset 1']['T'][0].groupby('time_counter.month').mean('time_counter') 
    
    # resample to give monthly means etc. 

    init_timer.append((datetime.now(),'xarray open_mfdataset connected'))

    #pdb.set_trace()
    # get lon, lat and time names from files
    check_var_name_present = True
    if do_LBC:check_var_name_present = False
    nav_lon_varname_dict,nav_lat_varname_dict,time_varname_dict,nav_lon_var_mat,nav_lat_var_mat,time_varname_mat = create_ncvar_lon_lat_time_dict(ncvar_d, gr_1st = gr_1st,check_var_name_present = check_var_name_present)
    #time_varname = time_varname_dict['Dataset 1']
    
    #pdb.set_trace()
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
   


    cutout_d = {}    
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:])
        cutout_d[th_d_ind] = {}
        cutout_d[th_d_ind]['cutxind'] = thd[th_d_ind]['cutx0'],thd[th_d_ind]['cutx1']
        cutout_d[th_d_ind]['cutyind'] = thd[th_d_ind]['cuty0'],thd[th_d_ind]['cuty1']
        cutout_d[th_d_ind]['do_cutout'] = False

        if cutout_d[th_d_ind]['cutxind'][0]!=0:cutout_d[th_d_ind]['do_cutout'] = True
        if cutout_d[th_d_ind]['cutyind'][0]!=0:cutout_d[th_d_ind]['do_cutout'] = True
        if cutout_d[th_d_ind]['cutxind'][1] is not None:cutout_d[th_d_ind]['do_cutout'] = True
        if cutout_d[th_d_ind]['cutyind'][1] is not None:
            cutout_d[th_d_ind]['do_cutout'] = True
            thd[th_d_ind]['y1'] = None
  

    print ('xarray open_mfdataset Finish',datetime.now())


    #pdb.set_trace()
    # Create lon and lat dictionaries
    #lon_d,lat_d = create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutxind,cutyind,cutout_data)
    lon_d,lat_d = create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutout_d,gr_1st = gr_1st)
    

    trim_lon_lat_with_thd = False
    for tmp_datstr in Dataset_lst:
        th_d_ind = int(tmp_datstr[8:])
        if (thd[th_d_ind]['lat0'] is not None)&(thd[th_d_ind]['lat1'] is not None)&\
           (thd[th_d_ind]['lon0'] is not None)&(thd[th_d_ind]['lon1'] is not None):
                #pdb.set_trace()

                thd_lon_lat_ind = (lon_d[th_d_ind]>thd[th_d_ind]['lon0']) & (lon_d[th_d_ind]<=thd[th_d_ind]['lon1']) & (lat_d[th_d_ind]>thd[th_d_ind]['lat0']) & (lat_d[th_d_ind]<=thd[th_d_ind]['lat1'])  


                thd[th_d_ind]['x0'] = np.where(thd_lon_lat_ind.any(axis = 0))[0].min()
                thd[th_d_ind]['x1'] = np.where(thd_lon_lat_ind.any(axis = 0))[0].max()
                thd[th_d_ind]['y0'] = np.where(thd_lon_lat_ind.any(axis = 1))[0].min()
                thd[th_d_ind]['y1'] = np.where(thd_lon_lat_ind.any(axis = 1))[0].max()

                trim_lon_lat_with_thd = True
    
    if trim_lon_lat_with_thd:
        lon_d,lat_d = create_lon_lat_dict(Dataset_lst,configd,thd,rootgrp_gdept_dict,xarr_dict,ncglamt,ncgphit,nav_lon_varname_dict,nav_lat_varname_dict,ncdim_d,cutout_d,gr_1st = gr_1st)
    

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

            
            xypos_dict[tmp_datstr]['XPOS_NN'] = griddata(points, values_X, (xypos_xmat, xypos_ymat), method='nearest')
            xypos_dict[tmp_datstr]['YPOS_NN'] = griddata(points, values_Y, (xypos_xmat, xypos_ymat), method='nearest')

            

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
    #grid_dict,nz = load_grid_dict(Dataset_lst,rootgrp_gdept_dict, thd, nce1t,nce2t,nce3t,configd, config_fnames_dict,cutxind,cutyind,cutout_data, do_mask_dict)
    grid_dict,nz = load_grid_dict(Dataset_lst,rootgrp_gdept_dict, thd, nce1t,nce2t,nce3t,configd, config_fnames_dict,cutout_d, do_mask_dict)
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
            #pdb.set_trace()
            var = var_d[1]['mat'][0]

    ## if var not in var_d[1]['mat']: var = var_d[1]['mat'][0]

    nice_varname_dict = {}
    for tmpvar in var_d[1]['mat']: nice_varname_dict[tmpvar] = tmpvar
    
    # add common variable names
    nice_varname_dict = add_default_nice_varname(nice_varname_dict)
    

    
    init_timer.append((datetime.now(),'Nice names loaded'))

    # extract time information from xarray.
    # needs to work for gregorian and 360 day calendars.
    # needs to work for as x values in a plot, or pcolormesh
    # needs work, xarray time is tricky

    init_timer.append((datetime.now(),'nc time started'))

    #pdb.set_trace()
    time_datetime,time_datetime_since_1970,ntime,ti, nctime_calendar_type = extract_time_from_xarr(xarr_dict['Dataset 1'][gr_1st],fname_dict['Dataset 1'][gr_1st][0],time_varname_dict['Dataset 1'][gr_1st],ncdim_d['Dataset 1'][gr_1st]['t'],date_in_ind,date_fmt,ti,verbose_debugging, curr_define_time_dict = define_time_dict['Dataset 1'][gr_1st])

    init_timer.append((datetime.now(),'nc time completed'))


    if justplot: 
        print('justplot:',justplot)
        print('Just plotting, and exiting, not interactive.')
        
        just_plt_cnt = 0
        njust_plt_cnt = 0

        if (justplot_date_ind is None)|(justplot_date_ind == 'None')|(justplot_date_ind == ''):
             
             #justplot_date_ind = time_datetime[ti].strftime(date_fmt)
             justplot_date_ind = ','.join(['%i'% ii for ii in range(ntime)])

        if (justplot_z_meth_zz is None)|(justplot_z_meth_zz == 'None'):
             justplot_z_meth_zz = 'ss:0,nb:0,df:0'

        if (justplot_secdataset_proc is None)|(justplot_secdataset_proc == 'None'):
             if nDataset == 1:
                justplot_secdataset_proc = 'Dataset_1'
             else:
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
                    just_plt_vals.append((secdataset_proc,justplot_date_ind_str, justplot_z_meth,justplot_zz, True, True, True, False, True, True))
                    njust_plt_cnt+=1
    init_timer.append((datetime.now(),'justplot prepared'))
    # repeat if comparing two time series. 




    # open file list with xarray
    for tmp_datstr in Dataset_lst: # xarr_dict.keys():
        #time_d[tmp_datstr] = {}
        for tmpgrid in xarr_dict[tmp_datstr].keys():
            (time_d[tmp_datstr][tmpgrid]['datetime'],time_d[tmp_datstr][tmpgrid]['datetime_since_1970'],tmp_ntime,tmp_ti, nctime_calendar_type) =  extract_time_from_xarr(xarr_dict[tmp_datstr][tmpgrid],fname_dict[tmp_datstr][tmpgrid][0], time_varname_dict[tmp_datstr][tmpgrid],ncdim_d[tmp_datstr][tmpgrid]['t'],date_in_ind,date_fmt,ti,verbose_debugging, curr_define_time_dict = define_time_dict[tmp_datstr][tmpgrid])
    print ('xarray start reading nctime',datetime.now())
    # add derived variables
    var_d,var_dim, var_grid = add_derived_vars(var_d,var_dim, var_grid, Dataset_lst)


    if var not in var_d[1]['mat']: var = var_d[1]['mat'][0]


    # add derived variables to nice names if mising. 

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

    #default regirdding settings
    regrid_meth = 2 # BL # 1 # NN
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

    if len(var_but_mat) == 0:
        print('No variable common to all datasets, len(var_but_mat) == 0')
        pdb.set_trace()
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

        if var_but_mat.size == 0:
            print('No common variable loaded across the Datasets, so nbutvar == 0')
            pdb.set_trace()


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


        if Obs_hide_edges is None: Obs_hide_edges = False
        if Obs_pair_loc is None: Obs_pair_loc = True
        if Obs_AbsAnom is None: Obs_AbsAnom = True

        #Obs_Type_argo = True
        #Obs_Type_ships = True
        #Obs_Type_drifter = True
        #Obs_Type_moored = True
        # 50 = ship, 53 = drifting buoy, 55 = moored buoy
        if Obs_Type_load_dict is None:
            Obs_Type_load_dict = {}
            Obs_Type_load_dict['show_with_diff_var'] = False

        Obs_Type_load_lst = obs_get_Obs_Type_load_lst()

        for tmp_Obs_Type_load in Obs_Type_load_lst:
            if tmp_Obs_Type_load not in Obs_Type_load_dict.keys():
                #Obs_Type_load_dict[tmp_Obs_Type_load] = True

                # if you're going to load all obs all the time, 
                # don't show them all
                if Obs_Type_load_dict['show_with_diff_var']:
                    Obs_Type_load_dict[tmp_Obs_Type_load] = False
                else:
                    Obs_Type_load_dict[tmp_Obs_Type_load] = True
    
    
        '''
        Obs_Type_load_dict['TS_argo'] = True
        Obs_Type_load_dict['TS_ships'] = True
        Obs_Type_load_dict['TS_gliders'] = True
        Obs_Type_load_dict['TS_other'] = True
        Obs_Type_load_dict['SST_ships'] = True
        Obs_Type_load_dict['SST_drifter'] = True
        Obs_Type_load_dict['SST_moored'] = True
        '''
        

        #Available Obs data types
        Obs_varlst = Obs_dict['Dataset 1'].keys()

        # Obs data types to hide
        #Obs_obstype_hide = ['ProfT','ProfS','SST_ins','SST_sat','SLA', 'ChlA']
        #Obs_obstype_hide = ['SST_sat']

        Obs_JULD_datetime_dict = Obs_setup_Obs_JULD_datetime_dict(Dataset_lst,Obs_varlst,Obs_dict)
   
        '''
        
        # a dictionary of Obs datetimes, assuming midday of each day. 
        Obs_JULD_datetime_dict = {}
        for tmp_datstr in Dataset_lst:
            Obs_JULD_datetime_dict[tmp_datstr] = {} #{'ProfT':Obs_dict[tmp_datstr]['ProfT']['JULD']}
            for ob_var in Obs_varlst:
                Obs_JULD_datetime_dict[tmp_datstr][ob_var] = Obs_dict[tmp_datstr][ob_var]['JULD']

        '''

        # Obs plotting options 
        Obs_scatEC = None
        # size of markers
        Obs_scatSS = matplotlib.rcParams['lines.markersize'] ** 2
        Obs_scatSS = 100 # matplotlib.rcParams['lines.markersize'] ** 2
        Obs_scatSS = 250 # matplotlib.rcParams['lines.markersize'] ** 2

        Obs_vis_d = Obs_setup_Obs_vis_d(Obs_varlst)
        '''
        
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

        '''
        # set up empty profile dictionaries for plotting
        (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
            obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst)

        # set reload Obs to true
        reload_Obs = True

        init_timer.append((datetime.now(),'Obs Loaded '))

    print('Creating Figure', datetime.now())
    ax = []

    fig_tit_str = 'Interactive figure, Select lat/lon in a); lon in b); lat  in c); depth in d) and time in e).\n'
    #if dataset_lab_d['Dataset 1'] is not None: fig_tit_str = fig_tit_str + ' Dataset 1 = %s;'%dataset_lab_d['Dataset 1']
    #if dataset_lab_d['Dataset 2'] is not None: fig_tit_str = fig_tit_str + ' Dataset 2 = %s;'%dataset_lab_d['Dataset 2']

    for tmp_datstr in Dataset_lst:
        #if dataset_lab_d[tmp_datstr] is not None: 
        fig_tit_str = fig_tit_str + ' %s = %s;'%(tmp_datstr,dataset_lab_d[tmp_datstr])


    fig_tit_str_int = 'Interactive figure, Select lat/lon in a); lon in b); lat  in c); depth in d) and time in e). %s[%i, %i, %i, %i] (thin = %i; thin_files = %i) '%(var,ii,jj,zz,ti, thd[1]['dx'], thd[1]['df'])
    fig_tit_str_lab = ''
    #if dataset_lab_d['Dataset 1'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 1 = %s;'%dataset_lab_d['Dataset 1']
    #if dataset_lab_d['Dataset 2'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 2 = %s;'%dataset_lab_d['Dataset 2']
    for tmp_datstr in Dataset_lst:
        #if dataset_lab_d[tmp_datstr] is not None: 
        fig_tit_str_lab = fig_tit_str_lab + ' %s = %s;'%(tmp_datstr,dataset_lab_d[tmp_datstr])
    fig_tit_str_lab = fig_tit_str_lab + ' Showing %s.'%(dataset_lab_d[tmp_datstr])

    nvarbutcol = 16 # 18
    nvarbutcol = 22 # 18
    nvarbutcol = 25 # 18


    if justplot:
        nvarbutcol = 1000


    do_z_spike_mag = True
    # dep
    zm_2d_meth_lst = ['zm','zx','zn','zs']
    zm_2d_meth_full_lst = ['Depth-Mean','Depth-Max','Depth-Min','Depth-Std']

    if do_z_spike_mag:
        zm_2d_meth_lst.append('zd')
        zm_2d_meth_full_lst.append('|Z-Spike|')

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
    wgap = 0.10
    wgap = 0.08
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
        tsaxtx_lst.append(ax[4].text(0.01,0.01,dataset_lab_d['Dataset 1'], ha = 'left', va = 'bottom', transform=ax[4].transAxes, color = 'r', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))

    elif nDataset ==2:
        tsaxtx_lst.append(ax[4].text(0.01,0.01,dataset_lab_d['Dataset 1'], ha = 'left', va = 'bottom', transform=ax[4].transAxes, color = 'r', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
        tsaxtx_lst.append(ax[4].text(0.99,0.01,dataset_lab_d['Dataset 2'], ha = 'right', va = 'bottom', transform=ax[4].transAxes, color = 'b', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))

        tsaxtxd_lst.append(ax[4].text(0.99,0.975,'Dat2-Dat1', ha = 'right', va = 'top', transform=ax[4].transAxes, color = 'g', fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
    
    else:
        tmp_vlist = np.linspace(0.85,0.15,nDataset)
        for tdsi,tmp_datstr in enumerate(Dataset_lst):
            tsaxtx_lst.append(ax[4].text(0.01,tmp_vlist[tdsi],dataset_lab_d[tmp_datstr], ha = 'left', va = 'top', transform=ax[4].transAxes, color = Dataset_col[tdsi], fontsize = 12,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none')))
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
        fig_tit_str_lab = fig_tit_str_lab + ' %s = %s;'%(tmp_datstr,dataset_lab_d[tmp_datstr])


    init_timer.append((datetime.now(),'Figure, adding text'))

    #flip depth axes
    for tmpax in ax[1:]: tmpax.invert_yaxis()
    #use log depth scale, setiched off as often causes problems (clashes with hidden axes etc).
    #for tmpax in ax[1:]: tmpax.set_yscale('log')

    # add hidden fill screen axes 
    clickax = fig.add_axes([0,0,1,1], frameon=False, zorder = -1)
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
        #pdb.set_trace()
        if var_dim[var_dat] == 3: #  2D (+time) vars 
            if   var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'T': tmpcol = 'deepskyblue'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'U': tmpcol = 'yellow'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'V': tmpcol = 'yellow'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'WW3': tmpcol = 'lightsteelblue'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'I': tmpcol = 'lightgreen'
        if var_dim[var_dat] == 4: #  3D (+time) vars 
            if   var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'T': tmpcol = 'b'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'U': tmpcol = 'gold'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'V': tmpcol = 'gold'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'WW3': tmpcol = 'navy'
            elif var_grid['Dataset 1'][var_dat][0].split('_')[0] == 'I': tmpcol = 'darkgreen'
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
                      'Contours','Grad','Time Diff','Sec Grid','TS Diag','LD time','Fcst Diag','Vis curr','MLD','Obs','Xsect','Time-Dist','Save Figure','Help','Quit'] 
    #'Reset zoom','Obs: sel','Obs: opt','Clim: Reset','Clim: Expand',
    
    Sec_regrid = False
    Sec_regrid_slice = False
    #Sec_regrid_slice = True
    if uniqconfig.size==1:
        func_names_lst.remove('Sec Grid')


    do_Xsect = True
    loaded_xsect = False
    if not do_Xsect:
        
        func_names_lst.remove('Xsect')
    else:
        figxs = None

    figtd = None


    
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
    #Time_Diff = True
    # For T Diff
    if ntime < 2: # no point being able to change lead time database if only one 
        func_names_lst.remove('Time Diff')
        do_Tdiff = False
    else:
        do_Tdiff = True

        if Time_Diff is None:            
            Time_Diff = False
        else:
            if Time_Diff:
                if ti == 0: ti=1 

        data_inst_Tm1 = {}
        #data_inst_Tm1['Dataset 1'],data_inst_Tm1['Dataset 2'] = None,None
        preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1 = 0.5,'None',0.5
        preload_data_ii_Tm1,preload_data_jj_Tm1 = 0,0
        preload_data_zz_Tm1 = 0

        

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
        tmpdataset_oper = '-'
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
    if   z_meth == 'z_slice':func_but_text_han['Depth level'].set_color('r')
    elif z_meth == 'ss':func_but_text_han['Surface'].set_color('r')
    elif z_meth == 'nb':func_but_text_han['Near-Bed'].set_color('r')
    elif z_meth == 'df':func_but_text_han['Surface-Bed'].set_color('r')
    elif z_meth == 'zm':func_but_text_han['Depth-Mean'].set_color('r')
    elif z_meth == 'zx':func_but_text_han['Depth-Mean'].set_color('r')
    elif z_meth == 'zn':func_but_text_han['Depth-Mean'].set_color('r')
    elif z_meth == 'zd':func_but_text_han['Depth-Mean'].set_color('r')
    elif z_meth == 'zs':func_but_text_han['Depth-Mean'].set_color('r')

    DepthMean_sw = 0

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
    func_but_line_han['Depth-Mean'][0].set_linewidth(1)
    func_but_line_han['Depth-Mean'][0].set_color('w')
    func_but_line_han['Depth-Mean'][0].set_path_effects(str_pe)
    func_but_line_han['Grad'][0].set_linewidth(1)
    func_but_line_han['Grad'][0].set_color('w')
    func_but_line_han['Grad'][0].set_path_effects(str_pe)
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

    #pdb.set_trace()
    but_text_han[var].set_color('r')

    if verbose_debugging: print('Added functions boxes', datetime.now())


    ###########################################################################
    # Define inner functions
    ###########################################################################


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
            #pdb.set_trace()
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

                # what do the local coordiantes of the click mean in terms of the data to plot.
                # if on the map, or the slices, need to covert from lon and lat to ii and jj, which is complex for amm15.
                # if in map, covert lon lat to ii,jj
                if ai == 0:
                    loni,latj= xlocval,ylocval

                    #print(sel_ii,sel_jj)
                    
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
                    # and reload slices, and hovmuller/time series

                elif ai in [1]: 
                    # if in ew slice, change ns slice, and hov/time series
                    loni= xlocval
                    sel_ii = (np.abs(ew_line_x - loni)).argmin()

                    #sel_zz = int( (1-normyloc)*clylim.ptp() + clylim.min() )
                    sel_zz = int( (1-normyloc)*np.ptp(clylim) + clylim.min() )
                    
                    
                elif ai in [2]:
                    # if in ns slice, change ew slice, and hov/time series
                    latj= xlocval
                    sel_jj = (np.abs(ns_line_y - latj)).argmin()
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
                    #print('No action for Profiles axes')
                    sel_zz = int( (1-normyloc)*np.ptp(clylim) + clylim.min() )
                else:
                    print('clicked in another axes??')
                    return
                    pdb.set_trace()


        # print('indices_from_ginput:',clii,cljj,sel_ax,sel_ii,sel_jj,sel_ti,sel_zz,xlocval,ylocval)
        # return sel_ax,sel_ii,sel_jj,sel_ti,sel_zz
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
            #if tmpdataset_oper == '-':
            #    secdataset_proc_figname = 'Diff_%s-%s'%(tmpdatasetnum_1,tmpdatasetnum_2)
            if tmpdataset_oper in tmpdataset_oper_lst:
                secdataset_proc_figname = 'Diff_%s%s%s'%(tmpdatasetnum_1,tmpdataset_oper,tmpdatasetnum_2)
        
        #print(do_grad)
        #pdb.set_trace()
        if resample_freq is None:
        #    fig_out_name = '%s/output_%s_%s_th%02i_fth%02i_i%04i_j%04i_t%03i_z%03i%s_g%1i_%s'%(      fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,secdataset_proc_figname)
            fig_out_name = '%s/output_%s_%s_th%02ifth%02ii%04ij%04it%03iz%03i%sg%1i_%s'%(      fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,secdataset_proc_figname)
        else:
        #    fig_out_name = '%s/output_%s_%s_th%02i_fth%02i_i%04i_j%04i_t%03i_z%03i%s_g%1i_res%s_%s'%(fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,resample_freq,secdataset_proc_figname)
            fig_out_name = '%s/output_%s_%s_th%02ifth%02ii%04ij%04it%03iz%03i%sg%1ir%s_%s'%(fig_dir,fig_lab,var,thd[1]['dx'],thd[1]['df'],ii,jj,ti,zz,z_meth,do_grad,resample_freq,secdataset_proc_figname)
        

        for tmp_datstr in Dataset_lst:
            #if dataset_lab_d[tmp_datstr] is not None: 
            fig_out_name = fig_out_name + '_d%s_%s'%(tmp_datstr[-1],dataset_lab_d[tmp_datstr])
        



        fig_tit_str_lab = ''
        if load_second_files == False:
            fig_tit_str_lab = dataset_lab_d['Dataset 1']
        else:
            if secdataset_proc in Dataset_lst:
                fig_tit_str_lab = '%s'%dataset_lab_d[secdataset_proc]
            else:
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]
                if tmpdataset_oper == '-':
                    fig_tit_str_lab = '%s minus %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
                elif tmpdataset_oper == '/':
                    fig_tit_str_lab = '%s over %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
                elif tmpdataset_oper == '%':
                    fig_tit_str_lab = '%s percent diff %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
        
        
        #fig.suptitle(fig_tit_str_int + '\n' + fig_tit_str_lab, fontsize=14)
        

        fig.suptitle( fig_tit_str_lab, fontsize=14)


        fig_out_name = fig_out_name.replace(' ','_')

        print(func_but_text_han.keys())
        print(func_but_line_han.keys())


        if fig_fname_lab is not None:
            fig_out_name = '%s_%s'%(fig_out_name,fig_fname_lab)

        #pdb.set_trace()

        for ss in func_but_line_han.keys():func_but_line_han[ss][0].set_visible(False)
        for ss in func_but_text_han.keys():func_but_text_han[ss].set_visible(False)

    
        if fig_cutout:
            fig_also_cutout_map_only = True


            bbox_cutout_pos = [[(but_x1+0.01), (0.066)],[(func_but_x0-0.01),0.965]]
                                     #            
            bbox_cutout_pos_inches = [[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.066)],[fig.get_figwidth()*(func_but_x0),fig.get_figheight()]]
            bbox_inches =  matplotlib.transforms.Bbox(bbox_cutout_pos_inches)
            
            if verbose_debugging: print('Save Figure: bbox_cutout_pos',bbox_cutout_pos, datetime.now())
            fig.savefig(fig_out_name+ '.png',bbox_inches = bbox_inches, dpi = figdpi)
            if fig_also_cutout_map_only:

                #half way between the left edge of the Map axes, and the hov axes
                map_x1_midpoint = (fig.axes[0].get_position().x1 + fig.axes[3].get_position().x0)/2 
                
                bbox_cutout_pos_inches_map_only = [[fig.get_figwidth()*(but_x1+0.01), fig.get_figheight()*(0.066)],[fig.get_figwidth()*(map_x1_midpoint),fig.get_figheight()*((fig.axes[0].get_position().y1+1)/2)]]
                bbox_inches_map_only =  matplotlib.transforms.Bbox(bbox_cutout_pos_inches_map_only)
                
                #if verbose_debugging: print('Save Figure: bbox_cutout_pos_',bbox_cutout_pos, datetime.now())
                fig.savefig(fig_out_name+ '_map_only.png',bbox_inches = bbox_inches_map_only, dpi = figdpi)
          

        else:
            fig.savefig(fig_out_name+ '.png', dpi = figdpi)


        for ss in func_but_text_han.keys():func_but_text_han[ss].set_visible(True)
        for ss in func_but_line_han.keys():func_but_line_han[ss][0].set_visible(True)


        #print('')
        #print(fig_out_name + '.png')
        #print('')





        fig.suptitle(fig_tit_str_int + '\n' + fig_tit_str_lab, fontsize=figsuptitfontsize)

        try:


            arg_output_text = 'flist1=$(echo "/dir1/file0[4-7]??_*.nc")\n'
            arg_output_text = arg_output_text + 'flist2=$(echo "/dir2/file0[4-7]??_*.nc")\n'
            arg_output_text = arg_output_text + '\n'
            arg_output_text = arg_output_text + "justplot_date_ind='%s,%s'\n"%(time_datetime[ti].strftime(date_fmt),time_datetime[ti].strftime(date_fmt))
            
            arg_output_text = arg_output_text + '\n\n\n'

            arg_output_text = arg_output_text + 'python NEMO_nc_slevel_viewer.py %s'%configd[1]
            arg_output_text = arg_output_text + ' "$flist1" '
            if zlim_max is not None:arg_output_text = arg_output_text + ' --zlim_max %i'%zlim_max
            arg_output_text = arg_output_text + ' --th 1 dxy %i'%thd[1]['dx']
            arg_output_text = arg_output_text + ' --th 1 df %i'%thd[1]['df']
            arg_output_text = arg_output_text + ' --datlab 1 %s'%dataset_lab_d['Dataset 1']
            arg_output_text = arg_output_text + ' --lon %f'%lon_d[1][jj,ii]
            arg_output_text = arg_output_text + ' --lat %f'%lat_d[1][jj,ii]
            if cur_xlim is not None: arg_output_text = arg_output_text + ' --xlim %f %f'%(cur_xlim[0],cur_xlim[1])
            if cur_ylim is not None: arg_output_text = arg_output_text + ' --ylim %f %f'%(cur_ylim[0],cur_ylim[1])
            arg_output_text = arg_output_text + ' --date_ind %s'%time_datetime[ti].strftime(date_fmt)
            arg_output_text = arg_output_text + ' --date_fmt %s'%date_fmt
            arg_output_text = arg_output_text + ' --var %s'%var
            #arg_output_text = arg_output_text + ' --z_meth %s'%z_meth
            arg_output_text = arg_output_text + ' --zz %s'%zz
            arg_output_text = arg_output_text + ' --do_grad %1i'%do_grad
            arg_output_text = arg_output_text + ' --clim_sym %s'%clim_sym
            arg_output_text = arg_output_text + ' --vis_curr %s'%vis_curr
            
            #if xlim is not None:arg_output_text = arg_output_text + ' --xlim %f %f'%tuple(xlim)
            #if ylim is not None:arg_output_text = arg_output_text + ' --ylim %f %f'%tuple(ylim)
            if load_second_files:
                #if configd[2] is not None: 
                arg_output_text = arg_output_text + ' --config_2nd %s'%configd[2]
                #arg_output_text = arg_output_text + ' --thin_2nd %i'%thd[2]['dx']
                arg_output_text = arg_output_text + ' --secdataset_proc "%s"'%secdataset_proc
                arg_output_text = arg_output_text + ' --fname_lst_2nd  "$flist2"'
                arg_output_text = arg_output_text + ' --clim_pair %s'%clim_pair

            arg_output_text = arg_output_text + ' --justplot_date_ind "$justplot_date_ind"'
            #arg_output_text = arg_output_text + " --justplot_date_ind '%s'"%time_datetime[ti].strftime(date_fmt)
            arg_output_text = arg_output_text + " --justplot_secdataset_proc '%s'"%justplot_secdataset_proc
            arg_output_text = arg_output_text + " --justplot_z_meth_zz '%s'"%justplot_z_meth_zz
            arg_output_text = arg_output_text + ' --justplot True'       
            arg_output_text = arg_output_text + '\n\n\n'       



            arg_output_text = arg_output_text + '\n\n\n'

            if zlim_max is not None:arg_output_text = arg_output_text + ' --zlim_max %i'%zlim_max
            arg_output_text = arg_output_text + ' --lon %f'%lon_d[1][jj,ii]
            arg_output_text = arg_output_text + ' --lat %f'%lat_d[1][jj,ii]
            if cur_xlim is not None: arg_output_text = arg_output_text + ' --xlim %f %f'%(cur_xlim[0],cur_xlim[1])
            if cur_ylim is not None: arg_output_text = arg_output_text + ' --ylim %f %f'%(cur_ylim[0],cur_ylim[1])
            arg_output_text = arg_output_text + ' --var %s'%var
            arg_output_text = arg_output_text + ' --zz %s'%zz
            arg_output_text = arg_output_text + ' --do_grad %1i'%do_grad
            arg_output_text = arg_output_text + ' --clim_sym %s'%clim_sym
            arg_output_text = arg_output_text + ' --vis_curr %s'%vis_curr
            
            arg_output_text = arg_output_text + ' --clim_pair %s'%clim_pair
            if clim is not None: arg_output_text = arg_output_text + ' --clim %s'%clim
            arg_output_text = arg_output_text + ' --Time_Diff %s'%Time_Diff
            if do_Obs:
                arg_output_text = arg_output_text + ' --Obs_hide_edges %s'%Obs_AbsAnom
                arg_output_text = arg_output_text + ' --Obs_pair_loc %s'%Obs_pair_loc
                arg_output_text = arg_output_text + ' --Obs_AbsAnom %s'%Obs_AbsAnom

                just_out_Obs_Type_load_str = ''
                for Obs_Type_var in Obs_Type_load_lst: 
                    if Obs_Type_load_dict[Obs_Type_var] == False:
                        just_out_Obs_Type_load_str =  '%s %s'%(just_out_Obs_Type_load_str,Obs_Type_var )
                if just_out_Obs_Type_load_str != '':
                    arg_output_text = arg_output_text + ' --Obs_type_hide%s'%just_out_Obs_Type_load_str



            arg_output_text = arg_output_text + ' --justplot_date_ind "$justplot_date_ind"'
            #arg_output_text = arg_output_text + " --justplot_date_ind '%s'"%time_datetime[ti].strftime(date_fmt)
            arg_output_text = arg_output_text + " --justplot_secdataset_proc '%s'"%justplot_secdataset_proc
            arg_output_text = arg_output_text + " --justplot_z_meth_zz '%s'"%justplot_z_meth_zz
            arg_output_text = arg_output_text + ' --justplot True' 


            fid = open(fig_out_name + '.txt','w')
            fid.write(arg_output_text)
            fid.close()
            
            print(' ')
            print(fig_out_name + '.png')
            print(fig_out_name + '.txt')
            if fig_also_cutout_map_only:
                print(fig_out_name + '_map_only.png')
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
        
        if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)

        #try:
        if True: 
            # extract plotting data (when needed), and subtract off difference files if necessary.

            if verbose_debugging: print('Set current data set (set of nc files) for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if verbose_debugging: print('Convert coordinates for config_2nd', datetime.now())
            

            iijj_ind = {}
            for tmp_datstr in Dataset_lst:
                th_d_ind = int(tmp_datstr[8:]) # int(tmp_datstr[-1])
                #iijj_ind[tmp_datstr] = None
                #if configd[th_d_ind] is not None:
                if configd[th_d_ind] !=  configd[1]:
                    


                    iijj_ind[tmp_datstr] = {}
                    
                    iijj_ind[tmp_datstr]['jj'], iijj_ind[tmp_datstr]['ii'] = regrid_params[tmp_datstr][3][jj,ii],regrid_params[tmp_datstr][4][jj,ii] # ind_from_lon_lat(tmp_datstr,configd,xypos_dict, lon_d,lat_d, thd,rot_dict,lon_d[1][jj,ii],lat_d[1][jj,ii])
                    
                    iijj_ind[tmp_datstr]['ew_jj'],iijj_ind[tmp_datstr]['ew_ii'] = regrid_params[tmp_datstr][3][jj,:],regrid_params[tmp_datstr][4][jj,:]
                    iijj_ind[tmp_datstr]['ew_bl_jj'],iijj_ind[tmp_datstr]['ew_bl_ii'] = regrid_params[tmp_datstr][0][:,jj,:],regrid_params[tmp_datstr][1][:,jj,:]
                    iijj_ind[tmp_datstr]['ew_wgt'] = regrid_params[tmp_datstr][2][:,jj,:]

                    iijj_ind[tmp_datstr]['ns_jj'],iijj_ind[tmp_datstr]['ns_ii'] = regrid_params[tmp_datstr][3][:,ii],regrid_params[tmp_datstr][4][:,ii]
                    iijj_ind[tmp_datstr]['ns_bl_jj'],iijj_ind[tmp_datstr]['ns_bl_ii'] = regrid_params[tmp_datstr][0][:,:,ii],regrid_params[tmp_datstr][1][:,:,ii]
                    iijj_ind[tmp_datstr]['ns_wgt'] = regrid_params[tmp_datstr][2][:,:,ii]

                        
                

            if verbose_debugging: print('Reload data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now())
            if verbose_debugging: print('Reload map, ew, ns, hov, ts',reload_map,reload_ew,reload_ns,reload_hov,reload_ts, datetime.now())
            prevtime = datetime.now()
            datstarttime = prevtime
            

            #to allow the time conversion between file sets with different times
            #if var == 'hs': pdb.set_trace()
            tmp_current_time = time_datetime[ti]
            #cur_time_datetime_dict = {}
            #pdb.set_trace()
            #for tmp_datstr in Dataset_lst:cur_time_datetime_dict[tmp_datstr] =  time_d[tmp_datstr][var_grid[tmp_datstr][var]]['datetime']
            #pdb.set_trace()
            time_datetime = time_d['Dataset 1'][var_grid['Dataset 1'][var][0]]['datetime']
            #time_datetime = cur_time_datetime_dict['Dataset 1']

            time_datetime_since_1970 = time_d['Dataset 1'][var_grid['Dataset 1'][var][0]]['datetime_since_1970']

            #print(tmp_current_time,time_datetime[0],time_datetime[-1],ti)
            if nctime_calendar_type in ['360_day','360']:
                #pdb.set_trace()
                ti = np.array([np.abs(ss *360*86400) for ss in (time_datetime - tmp_current_time)]).argmin()
            else:
                ti = np.array([np.abs(ss.total_seconds()) for ss in (time_datetime - tmp_current_time)]).argmin()
            #try:
            #    ti = np.array([np.abs(ss.total_seconds()) for ss in (time_datetime - tmp_current_time)]).argmin()
            #except:
            #    pdb.set_trace()
            #print(tmp_current_time,time_datetime[0],time_datetime[-1],ti)
            #print('\n\n\n\n\n')
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
                #pdb.set_trace()
                # if data_inst_1 is None (i.e. first loop), or
                #       if the time has changed, or
                #       if the variable has changed
                if  (data_inst is None)|(preload_data_ti != ti)|(preload_data_var != var)|(preload_data_ldi != ldi):

                    data_inst = None

                    if do_memory & do_timer: timer_lst.append(('Deleted data_inst',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    #pdb.set_trace()
                    #data_inst,psreload_data_ti,preload_data_var,preload_data_ldi= reload_data_instances(var,thd,ldi,ti,var_d,var_grid['Dataset 1'], xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files)
                    
                    data_inst,preload_data_ti,preload_data_var,preload_data_ldi= reload_data_instances_time(var,thd,ldi,ti,
                        time_datetime_since_1970[ti],time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files,
                        do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                    #pdb.set_trace()
                    if do_memory & do_timer: timer_lst.append(('Reloaded data_inst',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                    # For T Diff
                    if do_Tdiff:
                        #data_inst_Tm1['Dataset 1'],data_inst_Tm1['Dataset 2'] = None,None
                        #for tmp_datstr in Dataset_lst:data_inst_Tm1[tmp_datstr] = None
                        data_inst_Tm1 = None
                        data_inst_Tm1 = {}
                        ts_dat_dict_Tm1 = {}
                        hov_dat_dict_Tm1 = {}
                        for tmp_datstr in Dataset_lst:
                            data_inst_Tm1[tmp_datstr] = None     
                            ts_dat_dict_Tm1[tmp_datstr] = None     
                            hov_dat_dict_Tm1[tmp_datstr] = None          
                        preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1 = 0.5,'None',0.5
                        preload_data_ii_Tm1,preload_data_jj_Tm1 = 0,0
                        preload_data_zz_Tm1 = 0
                        Time_Diff_cnt = 0
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_Tm1',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

                if  (data_inst_U is None)|(preload_data_ti_U != ti)|(preload_data_ldi_U != ldi):
                    if vis_curr > 0:


                        data_inst_U = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_U',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_U,preload_data_ti_U,preload_data_var_U,preload_data_ldi_U = reload_data_instances_time(tmp_var_U,thd,ldi,ti,
                            time_datetime_since_1970[ti],time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files,
                            do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_U',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        
                        data_inst_V = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_V,preload_data_ti_V,preload_data_var_V,preload_data_ldi_V = reload_data_instances_time(tmp_var_V,thd,ldi,ti,
                            time_datetime_since_1970[ti],time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files,
                            do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_V',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))



                if  (data_inst_mld is None)|(preload_data_ti_mld != ti)|(preload_data_ldi_mld != ldi):
                    if reload_MLD:
                        data_inst_mld = None
                        if do_memory & do_timer: timer_lst.append(('Deleted data_inst_mld',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        data_inst_mld,preload_data_ti_mld,preload_data_var_mld,preload_data_ldi_mld= reload_data_instances_time(MLD_var,thd,ldi,ti,
                            time_datetime_since_1970[ti],time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files,
                            do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                        reload_MLD = False
                        if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_mld',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
            #pdb.set_trace()
            ###################################################################################################
            ### Status of buttons
            ###################################################################################################
            

            ###################################################################################################

            if reload_hov:
                Time_Diff_cnt_hovtime = 0
                if hov_time:
                    if var_dim[var] == 4:
                        #pdb.set_trace()
                        
                        hov_dat_dict = reload_hov_data_comb_time(var,var_d[1]['mat'],var_grid,var_dim,var_d['d'],ldi,thd, time_datetime,time_d, ii,jj,iijj_ind,nz,ntime, grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict,load_second_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)

                        if do_grad == 2:
                            hov_dat_dict = grad_vert_hov_prof_data(hov_dat_dict,
                                                                   meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx)
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
                    #ts_dat_dict = reload_ts_data_comb(var,var_dim,var_grid['Dataset 1'],ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,xarr_dict,do_mask_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,load_second_files)
                    ts_dat_dict = reload_ts_data_comb_time(var,var_dim,var_grid,ii,jj,iijj_ind,ldi,hov_dat_dict,time_datetime,time_d,z_meth,zz,zi,lon_d,lat_d,
                                                           xarr_dict,do_mask_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,
                                                           load_second_files,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                else:
                    ts_dat_dict['x'] = time_datetime
                    #ts_dat_dict['Dataset 1'] = np.ma.ones(ntime)*np.ma.masked
                    #ts_dat_dict['Dataset 2'] = np.ma.ones(ntime)*np.ma.masked
                    ts_dat_dict['Sec Grid'] = {}
                    for tmp_datstr in Dataset_lst:
                        ts_dat_dict[tmp_datstr] = np.ma.ones(ntime)*np.ma.masked
                        ts_dat_dict['Sec Grid'][tmp_datstr] = {}
                        ts_dat_dict['Sec Grid'][tmp_datstr]['x'] = time_datetime
                        ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = np.ma.ones(ntime)*np.ma.masked


                reload_ts = False

                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ts_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            
            ###################################################################################################




            if do_Tdiff:
                if ti == 0:
                    func_but_text_han['Time Diff'].set_color('0.5')
                else:
                    if Time_Diff:
                        func_but_text_han['Time Diff'].set_color('darkgreen')
                        
                        if (data_inst_Tm1['Dataset 1'] is None)|(preload_data_ti_Tm1 != (ti-1))|(preload_data_var_Tm1 != var)|(preload_data_ldi_Tm1 != ldi):

                            (data_inst_Tm1,preload_data_ti_Tm1,preload_data_var_Tm1,preload_data_ldi_Tm1) = reload_data_instances_time(var,thd,ldi,ti-1,  
                                time_datetime_since_1970[ti-1],time_d,var_d,var_grid, lon_d, lat_d, xarr_dict, grid_dict,var_dim,Dataset_lst,load_second_files,
                                do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                            # for tmp_datstr in Dataset_lst:  (data_inst[tmp_datstr] == data_inst_Tm1[tmp_datstr]).all()
                            if do_memory & do_timer: timer_lst.append(('Reloaded data_inst_Tm1',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                        #print('   Time_Diff_cnt_hovtime =',Time_Diff_cnt_hovtime)
                        #print('   preload_data_ii_Tm1',preload_data_ii_Tm1,ii,preload_data_jj_Tm1,jj,preload_data_zz_Tm1,zz)
                        if (hov_dat_dict_Tm1['Dataset 1'] is None)|(preload_data_ii_Tm1!=ii)|(preload_data_jj_Tm1!=jj)|(preload_data_zz_Tm1!=zz)|(preload_data_ti_Tm1 != (ti-1))|(preload_data_var_Tm1 != var)|(preload_data_ldi_Tm1 != ldi):
                            #print('   recalc Hov Diff')
                            
                            # can't just copy the dictionary, as the contents are still pointers
                            if hov_time:
                                ts_dat_dict_Tm1 = {}
                                hov_dat_dict_Tm1 = {}
                                for ss in ts_dat_dict.keys():ts_dat_dict_Tm1[ss] = ts_dat_dict[ss].copy()
                                for tmp_datstr in Dataset_lst:ts_dat_dict_Tm1['Sec Grid'] = {}
                                for tmp_datstr in Dataset_lst:ts_dat_dict_Tm1['Sec Grid'][tmp_datstr] = {}
                                for tmp_datstr in Dataset_lst:ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data'] = ts_dat_dict['Sec Grid'][tmp_datstr]['data'].copy()
                                #for tmp_datstr in Dataset_lst:id(ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data']) ,id( ts_dat_dict['Sec Grid'][tmp_datstr]['data'])


                                for ss in hov_dat_dict.keys():hov_dat_dict_Tm1[ss] = hov_dat_dict[ss].copy()
                                #pdb.set_trace()
                                #ts_dat_dict['Sec Grid'][tmp_datstr]['data']

                                for tmp_datstr in Dataset_lst:
                                    hov_dat_dict_Tm1[tmp_datstr][:,1:] = hov_dat_dict[tmp_datstr][:,:-1].copy()
                                    ts_dat_dict_Tm1[tmp_datstr][1:] = ts_dat_dict[tmp_datstr][:-1].copy()
                                    hov_dat_dict_Tm1[tmp_datstr][:,0] = np.ma.masked
                                    ts_dat_dict_Tm1[tmp_datstr][0] = np.ma.masked
                                    ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data'][1:] = ts_dat_dict['Sec Grid'][tmp_datstr]['data'][:-1].copy()
                                    ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data'][0] = np.ma.masked
                                
                                preload_data_ii_Tm1,preload_data_jj_Tm1 ,preload_data_zz_Tm1 = ii,jj,zz

                                reload_hov = True
                                reload_ts = True

                        if Time_Diff_cnt == 0:
                            for tmp_datstr in Dataset_lst:
                                data_inst[tmp_datstr] = data_inst[tmp_datstr] - data_inst_Tm1[tmp_datstr]

                            Time_Diff_cnt -= 1


                        if Time_Diff_cnt_hovtime == 0:
                            if hov_time:
                                for tmp_datstr in Dataset_lst:
                                    #print('   subtract hovtime_Tm1')
                                    hov_dat_dict[tmp_datstr] = hov_dat_dict[tmp_datstr] - hov_dat_dict_Tm1[tmp_datstr]
                                    ts_dat_dict[tmp_datstr] = ts_dat_dict[tmp_datstr] - ts_dat_dict_Tm1[tmp_datstr]
                                    ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = ts_dat_dict['Sec Grid'][tmp_datstr]['data'] - ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data']

                                Time_Diff_cnt_hovtime -= 1
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
                                for tmp_datstr in  Dataset_lst:
                                    data_inst[tmp_datstr] = data_inst[tmp_datstr] + data_inst_Tm1[tmp_datstr]
                                Time_Diff_cnt += 1

                            if Time_Diff_cnt_hovtime == -1:
                                for tmp_datstr in  Dataset_lst:
                                    if hov_time:
                                        hov_dat_dict[tmp_datstr] = hov_dat_dict[tmp_datstr] + hov_dat_dict_Tm1[tmp_datstr]
                                        ts_dat_dict[tmp_datstr] = ts_dat_dict[tmp_datstr] + ts_dat_dict_Tm1[tmp_datstr]
                                        ts_dat_dict['Sec Grid'][tmp_datstr]['data'] = ts_dat_dict['Sec Grid'][tmp_datstr]['data'] + ts_dat_dict_Tm1['Sec Grid'][tmp_datstr]['data']
                                Time_Diff_cnt_hovtime += 1

                            func_but_text_han['Clim: sym'].set_color('k')
                            clim_sym_but = 0
                            
                            reload_map = True
                            reload_ew = True
                            reload_ns = True
                            reload_hov = True
                            reload_ts = True

                            # clear the data_inst_Tm1 array if not in use
                            for tmp_datstr in  Dataset_lst:
                                data_inst_Tm1[tmp_datstr] = None
                                hov_dat_dict_Tm1[tmp_datstr] = None
                                ts_dat_dict_Tm1[tmp_datstr] = None

                if do_memory & do_timer: timer_lst.append(('Applied T_diff',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
                            


            for tmp_datstr in  Dataset_lst:
                if do_mask_dict[tmp_datstr]:
                    if var_dim[var] == 3:
                        data_inst[tmp_datstr] = np.ma.array(data_inst[tmp_datstr], mask = (grid_dict[tmp_datstr]['tmask'][0,:,:] == False))
                    elif var_dim[var] == 4:                        
                        data_inst[tmp_datstr] = np.ma.array(data_inst[tmp_datstr], mask = (grid_dict[tmp_datstr]['tmask'] == False))


            #pdb.set_trace()
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
                        map_dat_dict[tmp_datstr] = field_gradient_2d(map_dat_dict[tmp_datstr], thd[1]['dx']*grid_dict['Dataset 1']['e1t'],thd[1]['dx']*grid_dict['Dataset 1']['e2t'], do_mask = do_mask_dict['Dataset 1'], curr_griddict = grid_dict[tmp_datstr],
                                                                       meth_2d=grad_2d_meth,meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx) # scale up widths between grid boxes
                        
                        # if Sec_regrid is true, do gradient on _Sec_regrid in well, not instead, as needed for clim calcs
                        if Sec_regrid & (tmp_datstr!= 'Dataset 1'):
                            th_d_ind = int(tmp_datstr[8:])
                            #pdb.set_trace()
                            map_dat_dict[tmp_datstr + '_Sec_regrid'] = field_gradient_2d(map_dat_dict[tmp_datstr + '_Sec_regrid'], thd[th_d_ind]['dx']*grid_dict[tmp_datstr]['e1t'],thd[th_d_ind]['dx']*grid_dict[tmp_datstr]['e2t'], do_mask = do_mask_dict[tmp_datstr], curr_griddict = grid_dict[tmp_datstr],
                                                                                           meth_2d=grad_2d_meth,meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx) # scale up widths between grid boxes

                    
                    if do_memory & do_timer: timer_lst.append(('Calculated Grad of map_dat_dict',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

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
                    #mld_ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst_mld, lon_d[1], lat_d[1], grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
                    mld_ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst_mld, lon_d, lat_d, grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
                if reload_ew:
                    mld_ew_slice_dict = None
                    #mld_ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst_mld, lon_d[1], lat_d[1], grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
                    mld_ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst_mld, lon_d, lat_d, grid_dict, var_dim[MLD_var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                if do_memory & do_timer: timer_lst.append(('Reloaded MLD ns&ew slices',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
                    
            if reload_ew:
                if var_dim[var] == 4:
                    ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst, lon_d, lat_d, grid_dict, var_dim[var], regrid_meth,iijj_ind,Dataset_lst,configd)

                    if do_grad == 1:
                        ew_slice_dict = grad_horiz_ew_data(thd,grid_dict,jj, iijj_ind,ew_slice_dict,
                                                           meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, 
                                                           regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx,
                                                           grad_horiz_vert_wgt = grad_horiz_vert_wgt,Sec_regrid_slice = Sec_regrid_slice)
                    if do_grad == 2:
                        ew_slice_dict = grad_vert_ew_data(ew_slice_dict,
                                                          meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, 
                                                          regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx,
                                                          Sec_regrid_slice = Sec_regrid_slice)
                else:
                    ew_slice_dict = reload_ew_data_comb(ii,jj, data_inst, lon_d, lat_d, grid_dict, var_dim[var], regrid_meth,iijj_ind,Dataset_lst,configd)

                reload_ew = False

                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ew_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            

            if verbose_debugging: print('Reloaded  ew data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            if reload_ns:
                if var_dim[var] == 4:               
                    #ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
                    ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d, lat_d, grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                    if do_grad == 1:
                        ns_slice_dict = grad_horiz_ns_data(thd,grid_dict,ii, iijj_ind,ns_slice_dict,
                                                           meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, 
                                                           regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx,
                                                           grad_horiz_vert_wgt = grad_horiz_vert_wgt,Sec_regrid_slice = Sec_regrid_slice)
                    if do_grad == 2:
                        ns_slice_dict = grad_vert_ns_data(ns_slice_dict,
                                                          meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, 
                                                          regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx,
                                                          Sec_regrid_slice = Sec_regrid_slice)
                else:

                    #ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d[1], lat_d[1], grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
                    ns_slice_dict = reload_ns_data_comb(ii,jj, data_inst, lon_d, lat_d, grid_dict, var_dim[var],regrid_meth, iijj_ind,Dataset_lst,configd)
 
                  
                reload_ns = False
                
                if do_memory & do_timer: timer_lst.append(('Reloaded reload_ns_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))
            


            if verbose_debugging: print('Reloaded  ns data for ii = %s, jj = %s, zz = %s'%(ii,jj,zz), datetime.now(),'; dt = %s'%(datetime.now()-prevtime))
            prevtime = datetime.now()

            if profvis:
                #pdb.set_trace()
                pf_dat_dict = reload_pf_data_comb(data_inst,var,var_dim,ii,jj,nz,grid_dict,Dataset_lst,configd,iijj_ind)

                if do_grad == 2:
                    pf_dat_dict = grad_vert_hov_prof_data(pf_dat_dict,
                                                          meth=grad_meth, abs_pre = grad_abs_pre, abs_post = grad_abs_post, regrid_xy = grad_regrid_xy,dx_d_dx = grad_dx_d_dx)


                
                if do_memory & do_timer: timer_lst.append(('Reloaded reload_pf_data_comb',datetime.now(),psutil.Process(os.getpid()).memory_info().rss/1024/1024,))

            #if Obs and reloading,  
            if do_Obs:
                if reload_Obs:


                    Obs_dat_dict,Obs_var_lst_sub = Obs_reload_obs(var,Dataset_lst,tmp_current_time,ob_ti,Obs_dict,Obs_fname,Obs_JULD_datetime_dict,Obs_vis_d,Obs_varlst,Obs_reloadmeth,Obs_Type_load_dict)
                    '''
                    
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
                    '''

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
                    hov_dat = hov_dat_dict[secdataset_proc]
                    
                
                if Sec_regrid_slice:
                  #  pdb.set_trace()
                    ns_slice_dat = ns_slice_dict['Sec Grid'][secdataset_proc]['data']
                    ew_slice_dat = ew_slice_dict['Sec Grid'][secdataset_proc]['data']
                    ns_slice_x = ns_slice_dict['Sec Grid'][secdataset_proc]['x']
                    ew_slice_x = ew_slice_dict['Sec Grid'][secdataset_proc]['x']
                    ns_slice_y = ns_slice_dict['Sec Grid'][secdataset_proc]['y']
                    ew_slice_y = ew_slice_dict['Sec Grid'][secdataset_proc]['y']
                else:
                    ns_slice_dat = ns_slice_dict[secdataset_proc]
                    ew_slice_dat = ew_slice_dict[secdataset_proc]
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                ts_dat = ts_dat_dict[secdataset_proc]
                if vis_curr > 0:
                    map_dat_U = map_dat_dict_U[secdataset_proc]
                    map_dat_V = map_dat_dict_V[secdataset_proc]


                if do_MLD:
                    mld_ns_slice_dat = mld_ns_slice_dict[secdataset_proc]
                    mld_ew_slice_dat = mld_ew_slice_dict[secdataset_proc]
                    mld_ns_slice_x = mld_ns_slice_dict['x']
                    mld_ew_slice_x = mld_ew_slice_dict['x']
                    mld_ns_slice_y = mld_ns_slice_dict['y']
                    mld_ew_slice_y = mld_ew_slice_dict['y']
            else:
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]

                if tmpdataset_oper in tmpdataset_oper_lst:
                    map_dat = dataset_comp_func(map_dat_dict[tmpdataset_1], map_dat_dict[tmpdataset_2],method = tmpdataset_oper)
                    if var_dim[var] == 4:
                        #ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        #ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = dataset_comp_func(hov_dat_dict[tmpdataset_1], hov_dat_dict[tmpdataset_2],method = tmpdataset_oper)

                    #elif var_dim[var] == 3:
                    ns_slice_dat = dataset_comp_func(ns_slice_dict[tmpdataset_1], ns_slice_dict[tmpdataset_2],method = tmpdataset_oper)
                    ew_slice_dat = dataset_comp_func(ew_slice_dict[tmpdataset_1], ew_slice_dict[tmpdataset_2],method = tmpdataset_oper)
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                    ts_dat = dataset_comp_func(ts_dat_dict[tmpdataset_1], ts_dat_dict[tmpdataset_2],method = tmpdataset_oper)
                    if vis_curr > 0:
                        map_dat_U = dataset_comp_func(map_dat_dict_U[tmpdataset_1], map_dat_dict_U[tmpdataset_2],method = tmpdataset_oper)
                        map_dat_V = dataset_comp_func(map_dat_dict_V[tmpdataset_1], map_dat_dict_V[tmpdataset_2],method = tmpdataset_oper)
                else:
                    pdb.set_trace()

                """

                if tmpdataset_oper == '-':
                    map_dat = map_dat_dict[tmpdataset_1] - map_dat_dict[tmpdataset_2]
                    if var_dim[var] == 4:
                        #ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        #ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = hov_dat_dict[tmpdataset_1] - hov_dat_dict[tmpdataset_2]

                    #elif var_dim[var] == 3:
                    ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                    ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                    ts_dat = ts_dat_dict[tmpdataset_1] - ts_dat_dict[tmpdataset_2]
                    if vis_curr > 0:
                        map_dat_U = map_dat_dict_U[tmpdataset_1] - map_dat_dict_U[tmpdataset_2]
                        map_dat_V = map_dat_dict_V[tmpdataset_1] - map_dat_dict_V[tmpdataset_2]
                
                elif tmpdataset_oper == '/':
                    map_dat = map_dat_dict[tmpdataset_1] / map_dat_dict[tmpdataset_2]
                    if var_dim[var] == 4:
                        #ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        #ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = hov_dat_dict[tmpdataset_1] / hov_dat_dict[tmpdataset_2]

                    #elif var_dim[var] == 3:
                    ns_slice_dat = ns_slice_dict[tmpdataset_1] / ns_slice_dict[tmpdataset_2]
                    ew_slice_dat = ew_slice_dict[tmpdataset_1] / ew_slice_dict[tmpdataset_2]
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                    ts_dat = ts_dat_dict[tmpdataset_1] / ts_dat_dict[tmpdataset_2]
                    if vis_curr > 0:
                        map_dat_U = map_dat_dict_U[tmpdataset_1] / map_dat_dict_U[tmpdataset_2]
                        map_dat_V = map_dat_dict_V[tmpdataset_1] / map_dat_dict_V[tmpdataset_2]
                    '''
                elif tmpdataset_oper == '/':
                    map_dat = map_dat_dict[tmpdataset_1] / map_dat_dict[tmpdataset_2]
                    if var_dim[var] == 4:
                        #ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        #ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = hov_dat_dict[tmpdataset_1] / hov_dat_dict[tmpdataset_2]

                    #elif var_dim[var] == 3:
                    ns_slice_dat = (ns_slice_dict[tmpdataset_1] / ns_slice_dict[tmpdataset_2])
                    ew_slice_dat = ew_slice_dict[tmpdataset_1] / ew_slice_dict[tmpdataset_2]
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                    ts_dat = ts_dat_dict[tmpdataset_1] / ts_dat_dict[tmpdataset_2]
                    if vis_curr > 0:
                        map_dat_U = map_dat_dict_U[tmpdataset_1] / map_dat_dict_U[tmpdataset_2]
                        map_dat_V = map_dat_dict_V[tmpdataset_1] / map_dat_dict_V[tmpdataset_2]
                    '''
                elif tmpdataset_oper in ['/','%']:
                    map_dat = dataset_comp_func(map_dat_dict[tmpdataset_1], map_dat_dict[tmpdataset_2],method = tmpdataset_oper)
                    if var_dim[var] == 4:
                        #ns_slice_dat = ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2]
                        #ew_slice_dat = ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2]
                        #pdb.set_trace()
                        hov_dat = dataset_comp_func(hov_dat_dict[tmpdataset_1], hov_dat_dict[tmpdataset_2],method = tmpdataset_oper)

                    #elif var_dim[var] == 3:
                    ns_slice_dat = dataset_comp_func(ns_slice_dict[tmpdataset_1], ns_slice_dict[tmpdataset_2],method = tmpdataset_oper)
                    ew_slice_dat = dataset_comp_func(ew_slice_dict[tmpdataset_1], ew_slice_dict[tmpdataset_2],method = tmpdataset_oper)
                    ns_slice_x = ns_slice_dict['x']
                    ew_slice_x = ew_slice_dict['x']
                    ns_slice_y = ns_slice_dict['y']
                    ew_slice_y = ew_slice_dict['y']

                    ts_dat = dataset_comp_func(ts_dat_dict[tmpdataset_1], ts_dat_dict[tmpdataset_2],method = tmpdataset_oper)
                    if vis_curr > 0:
                        map_dat_U = dataset_comp_func(map_dat_dict_U[tmpdataset_1],map_dat_dict_U[tmpdataset_2],method = tmpdataset_oper)
                        map_dat_V = dataset_comp_func(map_dat_dict_V[tmpdataset_1], map_dat_dict_V[tmpdataset_2],method = tmpdataset_oper)
                else:
                    pdb.set_trace()
                """

                if do_MLD:  
                    mld_ns_slice_dat = mld_ns_slice_dict['Dataset 1'].copy()*0.
                    mld_ew_slice_dat = mld_ew_slice_dict['Dataset 1'].copy()*0.
                    mld_ns_slice_x = mld_ns_slice_dict['x']
                    mld_ew_slice_x = mld_ew_slice_dict['x']
                    mld_ns_slice_y = mld_ns_slice_dict['y']
                    mld_ew_slice_y = mld_ew_slice_dict['y']
                    
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
                    pdx = int(np.ceil(((ew_slice_x>cur_xlim[0]) &(ew_slice_x<cur_xlim[1]) ).sum()/pxy))

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
                pax.append(ax[1].pcolormesh(ew_slice_x[::pdx],ew_slice_y[:,::pdx],ew_slice_dat[:,::pdx],cmap = curr_cmap,norm = climnorm, rasterized = True))
                pax.append(ax[2].pcolormesh(ns_slice_x[::pdy],ns_slice_y[:,::pdy],ns_slice_dat[:,::pdy],cmap = curr_cmap,norm = climnorm, rasterized = True))
                pax.append(ax[3].pcolormesh(hov_dat_dict['x'],hov_dat_dict['y'],hov_dat,cmap = curr_cmap,norm = climnorm, rasterized = True))
            elif var_dim[var] == 3:

                if secdataset_proc in Dataset_lst:
                    
                    for dsi,tmp_datstr in enumerate(Dataset_lst):
                        tmplw = 0.5
                        if secdataset_proc == tmp_datstr:tmplw = 1
                        pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))
                        pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))

                        if do_ensemble:
                            pax2d.append(ax[1].plot(ew_slice_x,ens_ew_slice_dat[0],'k', lw = 1))
                            pax2d.append(ax[1].plot(ew_slice_x,ens_ew_slice_dat[1],'k', lw = 2))
                            pax2d.append(ax[1].plot(ew_slice_x,ens_ew_slice_dat[2],'k', lw = 1))
                            pax2d.append(ax[2].plot(ns_slice_x,ens_ns_slice_dat[0],'k', lw = 1))
                            pax2d.append(ax[2].plot(ns_slice_x,ens_ns_slice_dat[1],'k', lw = 2))
                            pax2d.append(ax[2].plot(ns_slice_x,ens_ns_slice_dat[2],'k', lw = 1))
                                
                else:
                    # only plot the current dataset difference
                    tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                    tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                    tmpdataset_oper = secdataset_proc[4]

                    if tmpdataset_oper in tmpdataset_oper_lst: 
                        

                        pax2d.append(ax[1].plot(ew_slice_x,dataset_comp_func(ew_slice_dict[tmpdataset_1],ew_slice_dict[tmpdataset_2], method = tmpdataset_oper),Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

                        pax2d.append(ax[2].plot(ns_slice_x,dataset_comp_func(ns_slice_dict[tmpdataset_1],ns_slice_dict[tmpdataset_2], method = tmpdataset_oper),Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

                        for tmp_datstr1 in Dataset_lst:
                            #th_d_ind1 = int(tmp_datstr1[-1])
                            th_d_ind1 = int(tmp_datstr1[8:])
                            for tmp_datstr2 in Dataset_lst:
                                #th_d_ind2 = int(tmp_datstr2[-1])
                                th_d_ind2 = int(tmp_datstr2[8:])
                                if tmp_datstr1!=tmp_datstr2:
                                    #tmp_diff_str_name = 'Dat%i-Dat%i'%(th_d_ind1,th_d_ind2) 
                                    tmp_diff_str_name = 'Dat%i%sDat%i'%(th_d_ind1,tmpdataset_oper,th_d_ind2)                               
                                    tmplw = 0.5
                                    if secdataset_proc == tmp_diff_str_name:tmplw = 1

                                    pax2d.append(ax[1].plot(ew_slice_x,dataset_comp_func(ew_slice_dict[tmp_datstr1], ew_slice_dict[tmp_datstr2],method = tmpdataset_oper),Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                    pax2d.append(ax[2].plot(ns_slice_x,dataset_comp_func(ns_slice_dict[tmp_datstr1], ns_slice_dict[tmp_datstr2],method = tmpdataset_oper),Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                                
                            pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))
                            pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))


                    """
                    if tmpdataset_oper == '-': 
                        

                        pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict[tmpdataset_1] - ew_slice_dict[tmpdataset_2],Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

                        pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict[tmpdataset_1] - ns_slice_dict[tmpdataset_2],Dataset_col_diff_dict[secdataset_proc]))
                        pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))

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

                                    pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict[tmp_datstr1] - ew_slice_dict[tmp_datstr2],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                    pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict[tmp_datstr1] - ns_slice_dict[tmp_datstr2],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))
                                                
                            pax2d.append(ax[1].plot(ew_slice_x,ew_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))
                            pax2d.append(ax[2].plot(ns_slice_x,ns_slice_dict['Dataset 1']*0, color = '0.5', ls = '--'))


                    """

            if do_MLD:
                #pdb.set_trace()
                if MLD_show:
                    #pdb.set_trace()
                    if var_dim[var]== 4:
                        mldax_lst.append(ax[1].plot(mld_ew_slice_x,mld_ew_slice_dat,'k', lw = 0.5))
                        mldax_lst.append(ax[2].plot(mld_ns_slice_x,mld_ns_slice_dat,'k', lw = 0.5))

            tsax_lst = []
            #Dataset_col = ['r','b','darkgreen','gold']
            # if Dataset X, plot all data sets
            if secdataset_proc in Dataset_lst:
                
                for dsi,tmp_datstr in enumerate(Dataset_lst):
                    tmplw = 0.5
                    if secdataset_proc == tmp_datstr:tmplw = 1
                    #tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))
                    tsax_lst.append(ax[4].plot(ts_dat_dict['Sec Grid'][tmp_datstr]['x'],ts_dat_dict['Sec Grid'][tmp_datstr]['data'],Dataset_col[dsi], lw = tmplw))


                    #tmp_ts_x = ts_dat_dict['x']
                    #tmp_ts_y = ts_dat_dict[tmp_datstr]
                    #tmp_ts_n = np.minimum(tmp_ts_x.size,tmp_ts_y.size)
                    #tsax_lst.append(ax[4].plot(ts_dat_dict['x'][:tmp_ts_n],ts_dat_dict[tmp_datstr][:tmp_ts_n],Dataset_col[dsi], lw = tmplw))

                    #tsax_lst.append(ax[4].plot(cur_time_datetime_dict[tmp_datstr],ts_dat_dict[tmp_datstr],Dataset_col[dsi], lw = tmplw))
                    if do_ensemble:
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[0],'k', lw = 1))
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[1],'k', lw = 1))
                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ens_ts_dat[2],'k', lw = 1))

                    
            else:
                # only plot the current dataset difference
                tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                tmpdataset_oper = secdataset_proc[4]
                if tmpdataset_oper in tmpdataset_oper_lst: 
                    
                    tsax_lst.append(ax[4].plot(ts_dat_dict['x'],dataset_comp_func(ts_dat_dict[tmpdataset_1], ts_dat_dict[tmpdataset_2], method = tmpdataset_oper),Dataset_col_diff_dict[secdataset_proc]))
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

                                tsax_lst.append(ax[4].plot(ts_dat_dict['x'],dataset_comp_func(ts_dat_dict[tmp_datstr1], ts_dat_dict[tmp_datstr2],method = tmpdataset_oper),Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))

                        tsax_lst.append(ax[4].plot(ts_dat_dict['x'],ts_dat_dict['Dataset 1']*0, color = '0.5', ls = '--'))




                else:
                    pdb.set_trace()
                """
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
                """

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


                        if z_meth == 'zd':
                            pf_xmean = pf_dat_dict[tmp_datstr].mean()
                            
                            tmpprof = pf_dat_dict[tmp_datstr]

                            tmpprof_1_hpf = tmpprof[1:-1] - ((tmpprof[0:-2] + 2*tmpprof[1:-1] + tmpprof[2:])/4)
                            
                            zzzwgt = np.ones((tmpprof_1_hpf.shape[0]))
                            zzzwgt[1::2] = -1
                            zd_ts_dat_1 = np.abs((tmpprof_1_hpf.T*zzzwgt).T.mean(axis = 0))
                            pfax_lst.append(ax[5].plot(pf_xmean*pf_dat_dict['y'][1:-1]/pf_dat_dict['y'][1:-1],pf_dat_dict['y'][1:-1],'k', lw = 0.25 ))
                            #pfax_lst.append(ax[5].plot(tmpprof_1_hpf - tmpprof_1_hpf.mean() + pf_xmean,pf_dat_dict['y'][1:-1],Dataset_col[dsi], lw = 0.5, ls = '--'))
                            #pfax_lst.append(ax[5].plot((tmpprof_1_hpf*zzzwgt) - (tmpprof_1_hpf*zzzwgt).mean() + pf_xmean,pf_dat_dict['y'][1:-1],Dataset_col[dsi], lw = 0.25))
                            pfax_lst.append(ax[5].plot(tmpprof_1_hpf + pf_xmean,pf_dat_dict['y'][1:-1],Dataset_col[dsi], lw = 0.5, ls = '--'))
                            pfax_lst.append(ax[5].plot((tmpprof_1_hpf*zzzwgt) + pf_xmean,pf_dat_dict['y'][1:-1],Dataset_col[dsi], lw = 0.25))
                            #pfax_lst.append(ax[5].plot(pf_dat_dict[tmp_datstr],pf_dat_dict['y'],Dataset_col[dsi], lw = tmplw))

                            del(tmpprof_1_hpf)

                        #else:
                        
                        # if Obs, plotted the observed data
                        if do_Obs:
                            if Obs_hide == False:
                            
                                '''# add obs data to pf_xvals to help choose y lims
                                for tmp_datstr in Dataset_lst:
                                    #if Obs_dat_dict[tmp_datstr][ob_var]['loaded']:
                                    for pfi in obs_obs_sel[tmp_datstr]:pf_xvals.append(pfi)
                                    for pfi in obs_mod_sel[tmp_datstr]:pf_xvals.append(pfi)
                                    '''
                                # add obs data to pf_xvals to help choose y lims
                                #if Obs_dat_dict[tmp_datstr][ob_var]['loaded']:
                                if obs_load_sel[secdataset_proc]:
                                    try:
                                        for pfi in obs_obs_sel[secdataset_proc]:pf_xvals.append(pfi)
                                        for pfi in obs_mod_sel[secdataset_proc]:pf_xvals.append(pfi)
                                    except:
                                        pdb.set_trace()
                                    #plot Obs profile
                                    if len(obs_obs_sel[secdataset_proc])==1:
                                        opax_lst.append([ax[5].axvline(obs_obs_sel[secdataset_proc],color = Obs_vis_d['Prof_obs_col'], ls = Obs_vis_d['Prof_obs_ls_2d'], lw = Obs_vis_d['Prof_obs_lw_2d'])])
                                        opax_lst.append([ax[5].axvline(obs_mod_sel[secdataset_proc],color = Obs_vis_d['Prof_mod_col'], ls = Obs_vis_d['Prof_mod_ls_2d'], lw = Obs_vis_d['Prof_mod_lw_2d'])])
                                        
                                    else:
                                        
                                        opax_lst.append(ax[5].plot(obs_obs_sel[secdataset_proc], obs_z_sel[secdataset_proc], color=Obs_vis_d['Prof_obs_col'], marker=Obs_vis_d['Prof_obs_ms'], linestyle=Obs_vis_d['Prof_obs_ls'], lw = Obs_vis_d['Prof_obs_lw']))
                                        opax_lst.append(ax[5].plot(obs_mod_sel[secdataset_proc], obs_z_sel[secdataset_proc], color=Obs_vis_d['Prof_mod_col'], marker=Obs_vis_d['Prof_mod_ms'], linestyle=Obs_vis_d['Prof_mod_ls'], lw = Obs_vis_d['Prof_mod_lw']))
                                        #pdb.set_trace()
                                    # give id info for Obs
                                    if obs_stat_type_sel[secdataset_proc] is not None:
                                        opaxtx_lst.append(ax[5].text(0.02,0.01,'ID = %s\nType: %i\n%s\n%s'%(obs_stat_id_sel[secdataset_proc] ,obs_stat_type_sel[secdataset_proc],obs_stat_time_sel[secdataset_proc].strftime('%Y/%m/%d %H:%M'), lon_lat_to_str(obs_lon_sel[secdataset_proc], obs_lat_sel[secdataset_proc])[0] ), ha = 'left', va = 'bottom', transform=ax[5].transAxes, color = 'k', fontsize = 10,bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none')))
                                        #opaxtx = ax[5].text(0.02,0.01,'ID = %s\nType: %i'%(obs_stat_id_sel[tmp_datstr] ,obs_stat_type_sel[tmp_datstr] ), ha = 'left', va = 'bottom', transform=ax[5].transAxes, color = 'k', fontsize = 10,bbox=dict(facecolor='white', alpha=0.75, pad=1, edgecolor='none'))

                        
                else:
                    # only plot the current dataset difference
                    tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                    tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                    tmpdataset_oper = secdataset_proc[4]

                    if tmpdataset_oper in tmpdataset_oper_lst: 
                        
                        #pf_xvals.append(pf_dat_dict[tmpdataset_1] - pf_dat_dict[tmpdataset_2])
                        for pfi in dataset_comp_func(pf_dat_dict[tmpdataset_1], pf_dat_dict[tmpdataset_2], method = tmpdataset_oper):pf_xvals.append(pfi)
                        pfax_lst.append(ax[5].plot(dataset_comp_func(pf_dat_dict[tmpdataset_1], pf_dat_dict[tmpdataset_2], method = tmpdataset_oper),pf_dat_dict['y'],Dataset_col_diff_dict[secdataset_proc]))
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
                                    for pfi in dataset_comp_func(pf_dat_dict[tmp_datstr1], pf_dat_dict[tmp_datstr2], method = tmpdataset_oper):pf_xvals.append(pfi)
                                    pfax_lst.append(ax[5].plot(dataset_comp_func(pf_dat_dict[tmp_datstr1], pf_dat_dict[tmp_datstr2], method = tmpdataset_oper),pf_dat_dict['y'],Dataset_col_diff_dict[tmp_diff_str_name], lw = tmplw))

                            pfax_lst.append(ax[5].plot(pf_dat_dict['Dataset 1']*0,pf_dat_dict['y'], color = '0.5', ls = '--'))



                    else:
                        pdb.set_trace()
                    """
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
                    """
                #pdb.set_trace()
                pf_xvals_min = np.ma.array(pf_xvals).ravel().min()
                pf_xvals_max = np.ma.array(pf_xvals).ravel().max()
                #pf_xvals_ptp = np.ma.array(pf_xvals).ravel().ptp()
                pf_xvals_ptp = np.ma.ptp(np.ma.array(pf_xvals).ravel())
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
            elif z_meth in zm_2d_meth_lst:
                for zm_2d_meth,zm_2d_meth_full in zip(zm_2d_meth_lst,zm_2d_meth_full_lst):
                    if z_meth == zm_2d_meth:
                        nice_lev = zm_2d_meth_full
            else:
                pdb.set_trace()

            #elif z_meth == 'zm':nice_lev = 'Depth-Mean'
            #elif z_meth == 'zx':nice_lev = 'Depth-Max'
            #elif z_meth == 'zn':nice_lev = 'Depth-Min'
            #elif z_meth == 'zd':nice_lev = 'Depth Spike Mag'
            #elif z_meth == 'zs':nice_lev = 'Depth-Std'

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
                tmpew_visible_ind = (ew_slice_x>=tmpew_xlim[0]) & (ew_slice_x<=tmpew_xlim[1]) 
                tmpns_visible_ind = (ns_slice_x>=tmpns_xlim[0]) & (ns_slice_x<=tmpns_xlim[1]) 

                tmp_ew_ylim = [0,zlim_min]
                tmp_ns_ylim = [0,zlim_min]
                if tmpew_visible_ind.any(): tmp_ew_ylim = [ew_slice_y[:,tmpew_visible_ind].max(),zlim_min]
                if tmpns_visible_ind.any(): tmp_ns_ylim = [ns_slice_y[:,tmpns_visible_ind].max(),zlim_min]
                tmp_hov_ylim = [hov_dat_dict['y'].max(),zlim_min]
                if var_dim[var] == 4:
                    ax[1].set_ylim(tmp_ew_ylim)
                    ax[2].set_ylim(tmp_ns_ylim)
                ax[3].set_ylim(tmp_hov_ylim)

                if profvis:
                    #pdb.set_trace()
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
                tmpew_visible_ind = (ew_slice_x>=tmpew_xlim[0]) & (ew_slice_x<=tmpew_xlim[1]) 
                tmpns_visible_ind = (ns_slice_x>=tmpns_xlim[0]) & (ns_slice_x<=tmpns_xlim[1]) 
                # catch edgecase where cross hairs don't pass through any water
                tmp_ew_slice_subset = ew_slice_dat[tmpew_visible_ind]
                tmp_ns_slice_subset = ns_slice_dat[tmpns_visible_ind]
                if (tmp_ew_slice_subset.size>0)&(not tmp_ew_slice_subset.mask.all()):
                    tmp_ew_ylim = np.array([tmp_ew_slice_subset.min(),tmp_ew_slice_subset.max()])
                    ax[1].set_ylim(tmp_ew_ylim)
                if (tmp_ns_slice_subset.size>0)&(not tmp_ns_slice_subset.mask.all()):
                    tmp_ns_ylim = np.array([tmp_ns_slice_subset.min(),tmp_ns_slice_subset.max()])
                    ax[2].set_ylim(tmp_ns_ylim)
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
                test_clim_code  = False

                if load_second_files & (clim_pair == True)&(secdataset_proc in Dataset_lst) :
                    if test_clim_code: print('load_second_files & (clim_pair == True)&(secdataset_proc in Dataset_lst)')
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
                        if test_clim_code: print('tmp_datstr')
                        
                        tmp_map_dat_clim = map_dat_dict[tmp_datstr][map_dat_reg_mask_1]
                        tmp_map_dat_clim = tmp_map_dat_clim[tmp_map_dat_clim.mask == False]

                        if len(tmp_map_dat_clim)>2:
                            tmp_map_dat_clim_lst.append(np.percentile(tmp_map_dat_clim,(5,95)))
                        
                        if test_clim_code: print('tmp_map_dat_clim_lst.append(np.percentile(tmp_map_dat_clim,(5,95)))')
                        
                        
                    tmp_map_dat_clim_mat = np.ma.array(tmp_map_dat_clim_lst).ravel()
                    if tmp_map_dat_clim_mat.size>1:
                        map_clim = np.ma.array([tmp_map_dat_clim_mat.min(),tmp_map_dat_clim_mat.max()])

                        if clim_sym: map_clim = np.ma.array([-1,1])*np.abs(map_clim).max()
                        if map_clim.mask.any() == False: set_clim_pcolor(map_clim, ax = ax[0])

                    
                    if test_clim_code: print('if map_clim.mask.any() == False: set_clim_pcolor(map_clim, ax = ax[0])')
                    # only apply to ns and ew slices, and hov if 3d variable. 

                    if var_dim[var] == 4:

                        if test_clim_code: print('if var_dim[var] == 4:')
                        '''
                        
                        ew_dat_reg_mask_1 = (ew_slice_x>tmpxlim[0]) & (ew_slice_x<tmpxlim[1]) 
                        ns_dat_reg_mask_1 = (ns_slice_x>tmpylim[0]) & (ns_slice_x<tmpylim[1])
                        
                        ns_slice_x = ns_slice_dict['x']
                        ew_slice_x = ew_slice_dict['x']
                        ns_slice_y = ns_slice_dict['y']
                        ew_slice_y = ew_slice_dict['y']
                        '''

                        # Not updated for Sec_regrid_slice
                        ew_dat_reg_mask_1 = (ew_slice_dict['x']>tmpxlim[0]) & (ew_slice_dict['x']<tmpxlim[1]) 
                        ns_dat_reg_mask_1 = (ns_slice_dict['x']>tmpylim[0]) & (ns_slice_dict['x']<tmpylim[1])
                        if test_clim_code: print("'ns_dat_reg_mask_1 = (ns_slice_dict['x']>tmpylim[0]) & (ns_slice_dict['x']<tmpylim[1])'")
                        
                        tmp_ew_dat_clim_lst,tmp_ns_dat_clim_lst, tmp_hov_dat_clim_lst = [],[],[]

                        for tmp_datstr in Dataset_lst:
                            if test_clim_code: print('for tmp_datstr in Dataset_lst:',tmp_datstr)

                            tmp_ew_slice_dict = ew_slice_dict[tmp_datstr]
                            tmp_ns_slice_dict = ns_slice_dict[tmp_datstr]
                            tmp_hov_dat_dict = hov_dat_dict[tmp_datstr]

                            if Sec_regrid_slice:

                                tmp_ew_slice_dict = ew_slice_dict['Sec Grid'][tmp_datstr]['data'].copy()
                                tmp_ns_slice_dict = ns_slice_dict['Sec Grid'][tmp_datstr]['data'].copy()
                                #tmp_hov_dat_dict = hov_dat_dict['Sec Grid'][tmp_datstr]['data'].copy()


                                #if tmp_datstr != Dataset_lst[0]:
                                ew_dat_reg_mask_1 = (ew_slice_dict['Sec Grid'][tmp_datstr]['x']>tmpxlim[0]) & (ew_slice_dict['Sec Grid'][tmp_datstr]['x']<tmpxlim[1]) 
                                ns_dat_reg_mask_1 = (ns_slice_dict['Sec Grid'][tmp_datstr]['x']>tmpylim[0]) & (ns_slice_dict['Sec Grid'][tmp_datstr]['x']<tmpylim[1])
                                #ew_slice_dict['Sec Grid'][tmp_datstr]['x']


                            #tmp_ew_dat_clim = ew_slice_dict[tmp_datstr][:,ew_dat_reg_mask_1]
                            #tmp_ns_dat_clim = ns_slice_dict[tmp_datstr][:,ns_dat_reg_mask_1]
                            #tmp_hov_dat_clim = hov_dat_dict[tmp_datstr].copy()

                            tmp_ew_dat_clim = tmp_ew_slice_dict[:,ew_dat_reg_mask_1]
                            tmp_ns_dat_clim = tmp_ns_slice_dict[:,ns_dat_reg_mask_1]
                            tmp_hov_dat_clim = tmp_hov_dat_dict.copy()
                            if test_clim_code: print('tmp_hov_dat_clim = hov_dat_dict[tmp_datstr].copy()',tmp_datstr)

                            tmp_ew_dat_clim = tmp_ew_dat_clim[tmp_ew_dat_clim.mask == False]
                            tmp_ns_dat_clim = tmp_ns_dat_clim[tmp_ns_dat_clim.mask == False]
                            tmp_hov_dat_clim = tmp_hov_dat_clim[tmp_hov_dat_clim.mask == False]
                            if test_clim_code: print('tmp_hov_dat_clim = tmp_hov_dat_clim[tmp_hov_dat_clim.mask == False]',tmp_datstr)


                            if len(tmp_ew_dat_clim)>2:   
                                tmp_ew_dat_clim_lst.append(np.percentile(tmp_ew_dat_clim,(5,95)))

                            if len(tmp_ns_dat_clim)>2:   
                                tmp_ns_dat_clim_lst.append(np.percentile(tmp_ns_dat_clim,(5,95)))

                            if len(tmp_hov_dat_clim)>2:  
                                tmp_hov_dat_clim_lst.append(np.percentile(tmp_hov_dat_clim,(5,95)))


                            if test_clim_code: print('tmp_hov_dat_clim_lst.append(np.percentile(tmp_hov_dat_clim,(5,95)))',tmp_datstr)


                        tmp_ew_dat_clim_mat =  np.ma.array(tmp_ew_dat_clim_lst).ravel()
                        tmp_ns_dat_clim_mat =  np.ma.array(tmp_ns_dat_clim_lst).ravel()
                        tmp_hov_dat_clim_mat = np.ma.array(tmp_hov_dat_clim_lst).ravel()
                        if test_clim_code: print('tmp_hov_dat_clim_mat')


                        if tmp_ew_dat_clim_mat.size>1:
                            ew_clim = np.ma.array([tmp_ew_dat_clim_mat.min(),tmp_ew_dat_clim_mat.max()])
                            if clim_sym: ew_clim = np.ma.array([-1,1])*np.abs(ew_clim).max()
                            if ew_clim.mask.any() == False: set_clim_pcolor(ew_clim, ax = ax[1])

                        if test_clim_code: print('ew_clim')

                        if tmp_ns_dat_clim_mat.size>1:
                            ns_clim = np.ma.array([tmp_ns_dat_clim_mat.min(),tmp_ns_dat_clim_mat.max()])
                            if clim_sym: ns_clim = np.ma.array([-1,1])*np.abs(ns_clim).max()
                            if ns_clim.mask.any() == False: set_clim_pcolor(ns_clim, ax = ax[2])

                        if test_clim_code: print('ns_clim')

                        if tmp_hov_dat_clim_mat.size>1:
                            hov_clim = np.ma.array([tmp_hov_dat_clim_mat.min(),tmp_hov_dat_clim_mat.max()])
                            if clim_sym: hov_clim = np.ma.array([-1,1])*np.abs(hov_clim).max()
                            if hov_clim.mask.any() == False: set_clim_pcolor(hov_clim, ax = ax[3])
                        if test_clim_code: print('hov_clim')
#
                            
                else:
                    if test_clim_code: print('else')
                    if (clim is None)| (secdataset_proc not in Dataset_lst):
                        if test_clim_code: print('(clim is None)| (secdataset_proc not in Dataset_lst):')
                        for tmpax in ax[:-1]:set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)
                        if test_clim_code: print('set_perc_clim_pcolor_in_region(5,95, ax = tmpax,sym = clim_sym)')
                        
                        
                    elif clim is not None:
                        if test_clim_code: print('elif clim is not None:')
                        if len(clim)>2:
                            for ai,tmpax in enumerate(ax):set_clim_pcolor((clim[2*ai:2*ai+1+1]), ax = tmpax)
                            set_clim_pcolor((clim[:2]), ax = ax[0])
                        if test_clim_code: print('set_clim_pcolor((clim[:2]), ax = ax[0])')
                        elif len(clim)==2:
                            for ai,tmpax in enumerate(ax):set_clim_pcolor((clim), ax = tmpax)
                            set_clim_pcolor((clim), ax = ax[0])
                        if test_clim_code: print('set_clim_pcolor((clim), ax = ax[0])')
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
            

            crshr_ax = []
            crshr_ax.append(ax[0].plot(lon_d[1][jj,:],lat_d[1][jj,:],color = '0.5', alpha = 0.5))
            crshr_ax.append(ax[0].plot(lon_d[1][:,ii],lat_d[1][:,ii],color = '0.5', alpha = 0.5))
            if Sec_regrid_slice:
                if secdataset_proc in Dataset_lst[1:]:
                    crshr_ax.append(ax[0].plot(ew_slice_dict['Sec Grid'][secdataset_proc]['lon'],ew_slice_dict['Sec Grid'][secdataset_proc]['lat'],color = '0.5', alpha = 0.5, ls = '--'))
                    crshr_ax.append(ax[0].plot(ns_slice_dict['Sec Grid'][secdataset_proc]['lon'],ns_slice_dict['Sec Grid'][secdataset_proc]['lat'],color = '0.5', alpha = 0.5, ls = '--'))
            
            '''
            cs_plot_1 = ax[0].plot(lon_d[1][jj,:],lat_d[1][jj,:],color = '0.5', alpha = 0.5) 
            cs_plot_2 = ax[0].plot(lon_d[1][:,ii],lat_d[1][:,ii],color = '0.5', alpha = 0.5)
            if Sec_regrid_slice:
                if secdataset_proc in Dataset_lst[1:]:
                    csrg_plot_1 = ax[0].plot(ew_slice_dict['Sec Grid']['Dataset 1']['lon'],ew_slice_dict['Sec Grid']['Dataset 1']['lat'],color = '0.5', alpha = 0.5, ls = '--') 
                    csrg_plot_2 = ax[0].plot(ns_slice_dict['Sec Grid']['Dataset 1']['lon'],ns_slice_dict['Sec Grid']['Dataset 1']['lat'],color = '0.5', alpha = 0.5, ls = '--')
            
            '''
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

            #if dataset_lab_d['Dataset 1']: tsaxtx1.set_text(dataset_lab_d['Dataset 1'])

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
                    conax.append(ax[1].contour(np.tile(ew_slice_x,(nz,1)),ew_slice_y,ew_slice_dat,cont_val_lst[1], colors = contcols, linewidths = contlws, alphas = contalphas))
                    conax.append(ax[2].contour(np.tile(ns_slice_x,(nz,1)),ns_slice_y,ns_slice_dat,cont_val_lst[2], colors = contcols, linewidths = contlws, alphas = contalphas))
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
                if (secdataset_proc in Dataset_lst):
                    # get current colorlimits
                    obs_clim = get_clim_pcolor(ax = ax[0])

                    
                    # for each Obs obs type, 
                    for obvi,ob_var in enumerate(Obs_var_lst_sub):
                        if Obs_dat_dict[secdataset_proc][ob_var]['loaded']:
                            
                            #extract the lon, lat, z and OBS
                            #pdb.set_trace()
                            tmpobsx = Obs_dat_dict[secdataset_proc][ob_var]['LONGITUDE']
                            tmpobsy = Obs_dat_dict[secdataset_proc][ob_var]['LATITUDE']
                            tmpobsz = Obs_dat_dict[secdataset_proc][ob_var]['DEPTH']

                            #if no obs actually loaded, skip.
                            if len(tmpobsx) == 0:
                                continue


                            if Obs_AbsAnom:
                                tmpobsdat_mat = Obs_dat_dict[secdataset_proc][ob_var]['OBS']
                            else:
                                tmpobsdat_mat = Obs_dat_dict[secdataset_proc][ob_var]['MOD_HX'] - Obs_dat_dict[secdataset_proc][ob_var]['OBS']

                            Obs_scatSS = Obs_vis_d['Scat_symsize'][ob_var] 
                            if Obs_hide_edges:
                                Obs_scatEC = None
                            else:
                                Obs_scatEC = Obs_vis_d['Scat_edgecol'][ob_var] 

                            # mask z levels where observations are masked. 
                            tmpobsz.mask = tmpobsz.mask |tmpobsdat_mat.mask



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
                                # find obs at this point
                                tmpobsdat = np.ma.array([tmpobsdat_mat[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                                
                                obs_dz_threshold = 50
                                if obs_dz_threshold is not None:

                                    # find distance to nearest level
                                    obs_obs_zi_dist_lst = np.ma.abs(tmpobsz - zz).min(axis = 1)
                                    # find depth at this point
                                    # tmpobsdat_zi = np.ma.array([tmpobsz[tmpzi,tmpzz] for tmpzi, tmpzz in enumerate(obs_obs_zi_lst)])
                                    # mask obs is outside the threshold
                                    tmpobsdat = np.ma.array(tmpobsdat,mask = ((tmpobsdat.mask) | (np.abs(obs_obs_zi_dist_lst)<np.abs(obs_dz_threshold)) == False))
                                    #pdb.set_trace()
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
                            elif z_meth in ['zm','zd']:
                                #depth mean obs if zm.
                                tmpobsdat = tmpobsdat_mat.mean(axis = 1)
                            elif z_meth == 'zx':
                                #depth mean obs if zm.
                                tmpobsdat = tmpobsdat_mat.max(axis = 1)
                            elif z_meth == 'zn':
                                #depth mean obs if zm.
                                tmpobsdat = tmpobsdat_mat.min(axis = 1)
                            elif z_meth == 'zs':
                                #depth mean obs if zm.
                                tmpobsdat = tmpobsdat_mat.std(axis = 1)
                            else:
                                pdb.set_trace()

                            # obs within the current maps
                            tmpobslatlonind = (tmpobsx>tmpxlim[0]) & (tmpobsx<tmpxlim[1]) & (tmpobsy>tmpylim[0]) & (tmpobsy<tmpylim[1]) 
                            tmp_obs_lst.append(tmpobsdat)
                            tmp_obs_llind_lst.append(tmpobslatlonind)
                            #scatter plot them
                            if Obs_hide == False:
                                if Obs_AbsAnom:
                                    oax_lst.append(ax[0].scatter(tmpobsx,tmpobsy,c = tmpobsdat, vmin = obs_clim[0],vmax = obs_clim[1], s = Obs_scatSS, edgecolors = Obs_scatEC ))
                                else:
                                    oax_lst.append(ax[0].scatter(tmpobsx,tmpobsy,c = tmpobsdat, s = Obs_scatSS, edgecolors = Obs_scatEC, cmap = matplotlib.cm.seismic ))
                                    
                    # if in anomaly mode, calculate the clim and colorbars for obs data. 
                    if (len(tmp_obs_lst)>0)& (Obs_AbsAnom==False) & (Obs_hide ==  False):
                        #try:

                        # join all obs types, and all the "within current map" TF array 
                        tmp_obs_mat=np.ma.concatenate(tmp_obs_lst)
                        tmp_obs_llind_mat=np.ma.concatenate(tmp_obs_llind_lst)

                        # if clim is sp ecified, use it, otherwise calculate it
                        if Obs_anom_clim is not None:
                            obs_OmB_clim = Obs_anom_clim
                        else:
                            #set obs_OmB_clim to None, so if not enough obs, doesn't crash when trying to set clims
                            obs_OmB_clim = None

                            '''
                            # if more than 3 values within the area, calc
                            if tmp_obs_llind_mat.sum()>3:
                                obs_OmB_clim = np.percentile(np.abs(tmp_obs_mat[(tmp_obs_mat.mask == False) & tmp_obs_llind_mat]),95)*np.array([-1,1])    
                            else:
                                obs_OmB_clim = np.percentile(np.abs(tmp_obs_mat[(tmp_obs_mat.mask == False)]),95)*np.array([-1,1])    
                            '''

                            # if more than 3 values within the area, calc 
                            if tmp_obs_llind_mat.sum()>3:
                                tmp_omb_val_clim = np.abs(tmp_obs_mat[(tmp_obs_mat.mask == False) & tmp_obs_llind_mat])
                            else:
                                tmp_omb_val_clim = np.abs(tmp_obs_mat[(tmp_obs_mat.mask == False)])
                            
                            # if more than 2 obs within region, calc clim values
                            if len(tmp_omb_val_clim)>2:
                                obs_OmB_clim = np.percentile(tmp_omb_val_clim,95)*np.array([-1,1])    
                            del(tmp_omb_val_clim)
                        #oax_lst[0].get_clim()
                        for tmp_oax_lst in oax_lst: tmp_oax_lst.set_clim(obs_OmB_clim)   
                        
                        if obs_OmB_clim is None:
                            #ensure clim is symetrical

                            #cycle through oax_lst and note clim
                            tmp_obs_OmB_clim_lst = []
                            for tmp_oax_lst in oax_lst:
                                tmp_obs_OmB_clim = tmp_oax_lst.get_clim()   
                                tmp_obs_OmB_clim_lst.append(tmp_obs_OmB_clim[0])
                                tmp_obs_OmB_clim_lst.append(tmp_obs_OmB_clim[1])

                            # convert this list to an array of abs values
                            tmp_obs_OmB_clim_mat = np.abs(tmp_obs_OmB_clim_lst)

                            # find max, and use this for a symetrical clim.
                            tmp_obs_OmB_clim = tmp_obs_OmB_clim_mat.max()*np.array([-1,1]) 
                            
                            # use this clim value
                            for tmp_oax_lst in oax_lst: tmp_oax_lst.set_clim(tmp_obs_OmB_clim)  

                            #delete temp arrays
                            del(tmp_obs_OmB_clim_lst) 
                            del(tmp_obs_OmB_clim_mat) 
                            del(tmp_obs_OmB_clim)         

                        #cbarobsax = [fig.add_axes([0.305,0.11, 0.15,0.02])]     
                        cbarobsax = [fig.add_axes([0.295,0.11, 0.15,0.02])]
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
                    #tmp_press = [[mouse_info['xdata'],mouse_info['ydata']]]
                    #print(tmp_press)
                    #tmp_press = [[mouse_info['x']/(fig.get_figwidth()*fig.get_dpi()),mouse_info['y']/(fig.get_figheight()*fig.get_dpi())]]
                    tmp_press = [[mouse_info['x']/(fig.get_window_extent().x1),mouse_info['y']/(fig.get_window_extent().y1)]]
                    #print(tmp_press)
                    

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
                if do_Obs:
                    reload_Obs = just_plt_vals[just_plt_cnt][9]
                try:
                    tmp_date_in_ind_ind = int(tmp_date_in_ind)
                except:
                    print('justplot_date_ind = %s'%tmp_date_in_ind)
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

            elif sel_ax in [3, 5]:
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
                    # load the obs for the correct varible and time
                    reload_Obs = True   

                    # deselect the current observation
                    (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                        obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst)
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
                                # load the obs for the correct varible and time
                                reload_Obs = True   

                                # deselect the current observation
                                (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst)
                                

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
                        zoom_corner_point = True

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
                                #tmpzoom0 = [[mouse_info['xdata'],mouse_info['ydata']]]
                                tmpzoom0 = [[mouse_info['x']/(fig.get_window_extent().x1),mouse_info['y']/(fig.get_window_extent().y1)]]
                                del(buttonpress)


                                if tmp_zoom_in:
                                    zoom_col = 'r'
                                else:
                                    zoom_col = 'g'
                                func_but_text_han['Zoom'].set_color(zoom_col)
                                # redraw canvas
                                fig.canvas.draw_idle()
                                
                                #flush canvas
                                fig.canvas.flush_events()
                                zoom0_ax,zoom0_ii,zoom0_jj,zoom0_ti,zoom0_zz,zoom0_sel_xlocval,zoom0_sel_ylocval = indices_from_ginput_ax(ax,tmpzoom0[0][0],tmpzoom0[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                if zoom0_ax in [1,2,3]:
                                    zlim_max = zoom0_zz
                                elif zoom0_ax in [0]:

                                    zoomax_lst = []


                                    if zoom_corner_point:
                                        if (zoom0_sel_xlocval is not None)&(zoom0_sel_ylocval is not None): # redundant, as (zoom0_ax in [0]) will be False is zoom0_ax = None
                                            zoomax_lst.append(ax[0].plot(zoom0_sel_xlocval,zoom0_sel_ylocval,'+', color = zoom_col))
                                            fig.canvas.draw_idle()
                                            if verbose_debugging: print('Canvas flush', datetime.now())
                                            fig.canvas.flush_events()
                                            if verbose_debugging: print('Canvas drawn and flushed', datetime.now())

                                    #tmpzoom1 = plt.ginput(1)
                                    buttonpress = True
                                    while buttonpress: buttonpress = plt.waitforbuttonpress()
                                    #tmpzoom1 = [[mouse_info['xdata'],mouse_info['ydata']]]
                                    tmpzoom1 = [[mouse_info['x']/(fig.get_window_extent().x1),mouse_info['y']/(fig.get_window_extent().y1)]]
                                    del(buttonpress)

                                    zoom1_ax,zoom1_ii,zoom1_jj,zoom1_ti,zoom1_zz, zoom1_sel_xlocval,zoom1_sel_ylocval = indices_from_ginput_ax(ax,tmpzoom1[0][0],tmpzoom1[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                        

                                    if zoom_corner_point:
                                        if (zoom1_sel_xlocval is not None)&(zoom1_sel_ylocval is not None):
                                            zoomax_lst.append(ax[0].plot(zoom1_sel_xlocval,zoom1_sel_ylocval,'+', color = zoom_col))
                                            fig.canvas.draw_idle()
                                            if verbose_debugging: print('Canvas flush', datetime.now())
                                            fig.canvas.flush_events()
                                            if verbose_debugging: print('Canvas drawn and flushed', datetime.now())

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
                                if zoom_corner_point:
                                    if zoom0_ax in [0]:     
                                        for zoomax in zoomax_lst:
                                            rem_loc = zoomax.pop(0)
                                            rem_loc.remove()          
                                                
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
                                    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst)
                                # select the observation with ginput
                                tmpobsloc = plt.ginput(1)

                                # convert to the nearest model grid box
                                # if fig zorder = -5
                                obs_jj,obs_ii = ind_from_lon_lat('Dataset 1',configd,xypos_dict, lon_d,lat_d, thd,rot_dict,tmpobsloc[0][0],tmpobsloc[0][1])
                                obs_ii = int(obs_ii)
                                obs_jj = int(obs_jj)
                                sel_xlocval, sel_ylocval = tmpobsloc[0][0],tmpobsloc[0][1]
                                obs_ax = 0
                                '''       
                                # if fig zorder = 0      
                                obs_ax,obs_ii,obs_jj,obs_ti,obs_zz, sel_xlocval,sel_ylocval = indices_from_ginput_ax(ax,tmpobsloc[0][0],tmpobsloc[0][1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                '''
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
                                    #    obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst, Fill = False)

                                    #pdb.set_trace()
                                    obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel,obs_load_sel = obs_load_selected_point(secdataset_proc, Dataset_lst, Obs_var_lst_sub, Obs_dat_dict, obs_lon, obs_lat)
                                    """
                                    tmpobs_dist_sel = {}
                                    (tmpobs_z_sel,tmpobs_obs_sel,tmpobs_mod_sel,tmpobs_lon_sel,tmpobs_lat_sel,
                                        tmpobs_stat_id_sel,tmpobs_stat_type_sel,tmpobs_stat_time_sel,obs_load_sel) = obs_reset_sel(Dataset_lst, Fill = False)

                                    



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

                                            tmpobs_z_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            tmpobs_obs_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            tmpobs_mod_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            tmpobs_lon_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            tmpobs_lat_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            #pdb.set_trace()
                                            tmpobs_stat_id_sel[ob_var][tmp_datstr] = ''
                                            tmpobs_stat_type_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked
                                            tmpobs_stat_time_sel[ob_var][tmp_datstr] = np.ma.zeros((1))*np.ma.masked

                                            tmpobs_dist_sel[ob_var][tmp_datstr] = tmp_obs_obs_dist



                                            if Obs_dat_dict[tmp_datstr][ob_var]['loaded']:
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

                                                tmpobs_dist_sel[ob_var][tmp_datstr] = tmp_obs_obs_dist
                                    '''
                                    obs_dist_sel_cf_dict = {}
                                    obs_dist_sel_cf_size_lst = []
                                    for tmp_datstr in Dataset_lst:
                                        obs_dist_sel_cf_dict[tmp_datstr] = np.array([tmpobs_dist_sel[ob_var][tmp_datstr] for ob_var in Obs_var_lst_sub])
                                        obs_dist_sel_cf_size_lst.append(obs_dist_sel_cf_dict[tmp_datstr].size)

                                    for tmp_datstr in Dataset_lst:
                                    
                                        # put all distances into one array
                                        obs_dist_sel_cf_mat = obs_dist_sel_cf_dict[tmp_datstr]
                                        #obs_dist_sel_cf_dict = {}
                                        #for tmp_datstr in Dataset_lst:obs_dist_sel_cf_dict[tmp_datstr] = np.array([tmpobs_dist_sel[ob_var][tmp_datstr] for ob_var in Obs_var_lst_sub])

                                        #if (obs_dist_sel_cf_mat).size>0:
                                        if (obs_dist_sel_cf_mat).size>0:
                                            
                                            # select the Obs Obs type closest to the selected point
                                            sel_Obs_var = Obs_var_lst_sub[obs_dist_sel_cf_mat.argmin()]
                                            
                                            #select obs data from Obs type closest to the selected point
                                            obs_z_sel[tmp_datstr] = tmpobs_z_sel[sel_Obs_var][tmp_datstr]
                                            obs_obs_sel[tmp_datstr] = tmpobs_obs_sel[sel_Obs_var][tmp_datstr]
                                            obs_mod_sel[tmp_datstr] = tmpobs_mod_sel[sel_Obs_var][tmp_datstr]
                                            obs_lon_sel[tmp_datstr] = tmpobs_lon_sel[sel_Obs_var][tmp_datstr]
                                            obs_lat_sel[tmp_datstr] = tmpobs_lat_sel[sel_Obs_var][tmp_datstr]

                                            obs_stat_id_sel[tmp_datstr] = tmpobs_stat_id_sel[sel_Obs_var][tmp_datstr]
                                            obs_stat_type_sel[tmp_datstr] = tmpobs_stat_type_sel[sel_Obs_var][tmp_datstr]
                                            obs_stat_time_sel[tmp_datstr] = tmpobs_stat_time_sel[sel_Obs_var][tmp_datstr]
                                        else:
                                            obs_z_sel[tmp_datstr] = np.ma.masked
                                            obs_obs_sel[tmp_datstr] = np.ma.masked
                                            obs_mod_sel[tmp_datstr] = np.ma.masked
                                            obs_lon_sel[tmp_datstr] = np.ma.masked
                                            obs_lat_sel[tmp_datstr] = np.ma.masked

                                            obs_stat_id_sel[tmp_datstr] = ''
                                            obs_stat_type_sel[tmp_datstr] = None
                                            obs_stat_time_sel[tmp_datstr] = ''
                             

                                    del(tmpobs_z_sel)
                                    del(tmpobs_obs_sel)
                                    del(tmpobs_mod_sel)
                                    del(tmpobs_lon_sel)
                                    del(tmpobs_lat_sel)
                                    del(tmpobs_dist_sel)

                                    del(tmpobs_stat_id_sel)
                                    del(tmpobs_stat_type_sel)
                                    del(tmpobs_stat_time_sel)


                                            ## append this data to an array to help select x and y lims
                                            #for tmp_datstr in Dataset_lst:
                                            #    tmp_pf_ylim_dat = np.ma.append(np.array(tmp_py_ylim),obs_z_sel[tmp_datstr])
                                            #    #tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel)[tmp_datstr],obs_obs_sel)
                                            #    tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel[tmp_datstr]),obs_obs_sel[tmp_datstr])
                                            
                                        #else:
                                    if (np.array(obs_dist_sel_cf_size_lst) == 0).all():
                                        
                                        (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                            obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst)


                                    '''
                                    #if Obs_dat_dict[secdataset_proc][ob_var]['loaded']:
                                    #pdb.set_trace()
                                    # put all distances into one array
                                    #obs_dist_sel_cf_mat = np.array([tmpobs_dist_sel[ob_var][secdataset_proc] for ob_var in Obs_var_lst_sub])
                                    obs_dist_sel_cf_lst = []
                                    for ob_var in Obs_var_lst_sub:
                                        if Obs_dat_dict[secdataset_proc][ob_var]['loaded']:
                                            obs_dist_sel_cf_lst.append(tmpobs_dist_sel[ob_var][secdataset_proc])
                                        else:
                                            obs_dist_sel_cf_lst.append(np.ma.masked)
                                    obs_dist_sel_cf_mat = np.ma.array(obs_dist_sel_cf_lst)

                                    #obs_dist_sel_cf_dict = {}
                                    #for tmp_datstr in Dataset_lst:obs_dist_sel_cf_dict[tmp_datstr] = np.array([tmpobs_dist_sel[ob_var][tmp_datstr] for ob_var in Obs_var_lst_sub])

                                    if ((obs_dist_sel_cf_mat).size>0) & (~obs_dist_sel_cf_mat.mask.any()):
                                        
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

                                        '''
                                        pdb.set_trace()
                                        # append this data to an array to help select x and y lims
                                        for tmp_datstr in Dataset_lst:
                                            tmp_pf_ylim_dat = np.ma.append(np.array(tmp_py_ylim),obs_z_sel[tmp_datstr])
                                            #tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel)[tmp_datstr],obs_obs_sel)
                                            tmp_pf_xlim_dat = np.ma.append(np.ma.append(np.array(pf_xlim),obs_mod_sel[tmp_datstr]),obs_obs_sel[tmp_datstr])
                                        '''
                                        
                                    else:
                                        
                                        (obs_z_sel,obs_obs_sel,obs_mod_sel,obs_lon_sel,obs_lat_sel,
                                            obs_stat_id_sel,obs_stat_type_sel,obs_stat_time_sel) = obs_reset_sel(Dataset_lst)
                                    """
                            elif mouse_info['button'].name == 'RIGHT':

                                # Bring up a options window for Obs
                                
                                # button names 
                                #

                                obs_but_names = [ss for ss in Obs_vis_d['visible'].keys()]
                                obs_but_names = obs_but_names + ['Hide_Obs','Edges','Loc','AbsAnom','Obs_show_with_diff_var',
                                                 #'Obs_Type_TSargo','Obs_Type_TSships','Obs_Type_TSgliders','Obs_Type_TSother',
                                                 'Obs_Type_T_argo','Obs_Type_T_ships','Obs_Type_T_gliders','Obs_Type_T_other',
                                                 'Obs_Type_S_argo','Obs_Type_S_ships','Obs_Type_S_gliders','Obs_Type_S_other',
                                                 'Obs_Type_SSTships','Obs_Type_SSTdrifter','Obs_Type_SSTmoored',
                                                 'Close']
                                
                                
                                '''
                                obs_but_names = ['ProfT','SST_ins','SST_sat','ProfS','SLA','ChlA',
                                                 'Hide_Obs','Edges','Loc','AbsAnom','Obs_show_with_diff_var',
                                                 #'Obs_Type_TSargo','Obs_Type_TSships','Obs_Type_TSgliders','Obs_Type_TSother',
                                                 'Obs_Type_T_argo','Obs_Type_T_ships','Obs_Type_T_gliders','Obs_Type_T_other',
                                                 'Obs_Type_S_argo','Obs_Type_S_ships','Obs_Type_S_gliders','Obs_Type_S_other',
                                                 'Obs_Type_SSTships','Obs_Type_SSTdrifter','Obs_Type_SSTmoored',
                                                 'Close']
                                '''
                                
                                
                                # button switches  
                                obs_but_sw = {}
                                obs_but_sw['Hide_Obs'] = {'v':Obs_hide, 'T':'Show Obs','F': 'Hide Obs'}
                                #obs_but_sw['Edges'] = {'v':Obs_hide, 'T':'Show Edges','F': 'Hide Edges'}
                                #obs_but_sw['Loc'] = {'v':Obs_hide, 'T':"Don't Selected point",'F': 'Move Selected point'}
                                obs_but_sw['Edges'] = {'v':Obs_hide_edges, 'T':'Show Edges','F': 'Hide Edges'}
                                obs_but_sw['Loc'] = {'v':Obs_pair_loc, 'T':"Don't Selected point",'F': 'Move Selected point'}
                                obs_but_sw['AbsAnom'] = {'v':Obs_AbsAnom, 'T':"Observed Values",'F': 'Model - Obs'}
                                obs_but_sw['Obs_show_with_diff_var'] = {'v':Obs_Type_load_dict['show_with_diff_var'] , 'T':"No obs mixing",'F': 'Allow obs mixing'}
                                #obs_but_sw['Obs_Type_argo'] = {'v':Obs_Type_argo, 'T':"Show Argo TS",'F': 'Hide Argo TS'}
                                #obs_but_sw['Obs_Type_ships'] = {'v':Obs_Type_ships, 'T':"Show ships SST",'F': 'Hide ships SST'}
                                #obs_but_sw['Obs_Type_drifter'] = {'v':Obs_Type_drifter, 'T':"Show drifter SST",'F': 'Hide drifter SST'}
                                #obs_but_sw['Obs_Type_moored'] = {'v':Obs_Type_moored, 'T':"Show moored SST",'F': 'Hide moored SST'}
                                #obs_but_sw['Obs_Type_TSargo'] = {'v':Obs_Type_load_dict['TS_argo'], 'T':"Show Argo TS",'F': 'Hide Argo TS'}
                                #obs_but_sw['Obs_Type_TSships'] = {'v':Obs_Type_load_dict['TS_ships'], 'T':"Show Ships TS",'F': 'Hide Ships TS'}
                                #obs_but_sw['Obs_Type_TSgliders'] = {'v':Obs_Type_load_dict['TS_gliders'], 'T':"Show Glider TS",'F': 'Hide Glider TS'}
                                #obs_but_sw['Obs_Type_TSother'] = {'v':Obs_Type_load_dict['TS_other'], 'T':"Show Other TS",'F': 'Hide Other TS'}
                                
                                obs_but_sw['Obs_Type_T_argo'] = {'v':Obs_Type_load_dict['T_argo'], 'T':"Hide Argo T",'F': 'Show Argo T','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_T_ships'] = {'v':Obs_Type_load_dict['T_ships'], 'T':"Hide Ships T",'F': 'Show Ships T','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_T_gliders'] = {'v':Obs_Type_load_dict['T_gliders'], 'T':"Hide Glider T",'F': 'Show Glider T','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_T_other'] = {'v':Obs_Type_load_dict['T_other'], 'T':"Hide Other T",'F': 'Show Other T','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_S_argo'] = {'v':Obs_Type_load_dict['S_argo'], 'T':"Hide Argo S",'F': 'Show Argo S','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_S_ships'] = {'v':Obs_Type_load_dict['S_ships'], 'T':"Hide Ships S",'F': 'Show Ships S','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_S_gliders'] = {'v':Obs_Type_load_dict['S_gliders'], 'T':"Hide Glider S",'F': 'Show Glider S','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_S_other'] = {'v':Obs_Type_load_dict['S_other'], 'T':"Hide Other S",'F': 'Show Other S','T_col':'k','F_col':'0.5'}
                                
                                
                                obs_but_sw['Obs_Type_SSTships'] = {'v':Obs_Type_load_dict['SST_ships'], 'T':"Hide ships SST",'F': 'Show ships SST','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_SSTdrifter'] = {'v':Obs_Type_load_dict['SST_drifter'], 'T':"Hide drifter SST",'F': 'Show drifter SST','T_col':'k','F_col':'0.5'}
                                obs_but_sw['Obs_Type_SSTmoored'] = {'v':Obs_Type_load_dict['SST_moored'], 'T':"Hide moored SST",'F': 'Show moored SST','T_col':'k','F_col':'0.5'}
                                
                                # Add obs variable type
                                for ob_var in Obs_var_lst_sub:  obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var,'T_col':'k','F_col':'0.5'}
                                
                                # Add all obs types
                                for ob_var in Obs_varlst:       obs_but_sw[ob_var] = {'v':Obs_vis_d['visible'][ob_var] , 'T':ob_var,'F': ob_var,'T_col':'k','F_col':'0.5'}

                                obbut_sel = pop_up_opt_window(obs_but_names, opt_but_sw = obs_but_sw)


                                # Set the main figure and axis to be current
                                plt.figure(fig.figure)
                                plt.sca(clickax)

            
                                # if the button closed was one of the Obs types, add or remove from the hide list
                                for ob_var in ['ProfT','SST_ins','SST_sat','ProfS','SLA','ChlA']:
                                    if obbut_sel == ob_var:
                                        #if ob_var in Obs_vis_d['visible']:.keys()
                                        Obs_vis_d['visible'][ob_var] = not Obs_vis_d['visible'][ob_var] 
                                # if the button closed was one of the Obs types, add or remove from the hide list
                                            
                                if obbut_sel == 'Hide_Obs': Obs_hide = not Obs_hide
                                if obbut_sel == 'Edges':    Obs_hide_edges = not Obs_hide_edges
                                if obbut_sel == 'Loc':      Obs_pair_loc = not Obs_pair_loc
                                if obbut_sel == 'AbsAnom':      Obs_AbsAnom = not Obs_AbsAnom
                                #if obbut_sel == 'Obs_Type_argo':    Obs_Type_argo    = not Obs_Type_argo
                                #if obbut_sel == 'Obs_Type_ships':   Obs_Type_ships   = not Obs_Type_ships
                                #if obbut_sel == 'Obs_Type_drifter': Obs_Type_drifter = not Obs_Type_drifter
                                #if obbut_sel == 'Obs_Type_moored':  Obs_Type_moored  = not Obs_Type_moored

                                #if obbut_sel == 'Obs_Type_TSargo':    Obs_Type_load_dict['TS_argo']    = not Obs_Type_load_dict['TS_argo']
                                #if obbut_sel == 'Obs_Type_TSships':    Obs_Type_load_dict['TS_ships']    = not Obs_Type_load_dict['TS_ships']
                                #if obbut_sel == 'Obs_Type_TSgliders':    Obs_Type_load_dict['TS_gliders']    = not Obs_Type_load_dict['TS_gliders']
                                #if obbut_sel == 'Obs_Type_TSother':    Obs_Type_load_dict['TS_other']    = not Obs_Type_load_dict['TS_other']
                                if obbut_sel == 'Obs_Type_S_argo':    Obs_Type_load_dict['S_argo']    = not Obs_Type_load_dict['S_argo']
                                if obbut_sel == 'Obs_Type_S_ships':    Obs_Type_load_dict['S_ships']    = not Obs_Type_load_dict['S_ships']
                                if obbut_sel == 'Obs_Type_S_gliders':    Obs_Type_load_dict['S_gliders']    = not Obs_Type_load_dict['S_gliders']
                                if obbut_sel == 'Obs_Type_S_other':    Obs_Type_load_dict['S_other']    = not Obs_Type_load_dict['S_other']
                                
                                if obbut_sel == 'Obs_Type_T_argo':    Obs_Type_load_dict['T_argo']    = not Obs_Type_load_dict['T_argo']
                                if obbut_sel == 'Obs_Type_T_ships':    Obs_Type_load_dict['T_ships']    = not Obs_Type_load_dict['T_ships']
                                if obbut_sel == 'Obs_Type_T_gliders':    Obs_Type_load_dict['T_gliders']    = not Obs_Type_load_dict['T_gliders']
                                if obbut_sel == 'Obs_Type_T_other':    Obs_Type_load_dict['T_other']    = not Obs_Type_load_dict['T_other']
                                
                                if obbut_sel == 'Obs_Type_SSTships':   Obs_Type_load_dict['SST_ships']    = not Obs_Type_load_dict['SST_ships']
                                if obbut_sel == 'Obs_Type_SSTdrifter': Obs_Type_load_dict['SST_drifter']    = not Obs_Type_load_dict['SST_drifter']
                                if obbut_sel == 'Obs_Type_SSTmoored':  Obs_Type_load_dict['SST_moored']    = not Obs_Type_load_dict['SST_moored']
                                if obbut_sel == 'Obs_show_with_diff_var':  Obs_Type_load_dict['show_with_diff_var']     = not Obs_Type_load_dict['show_with_diff_var'] 


                                reload_Obs = True
                        
                        elif but_name == 'Time-Dist':
                            if secdataset_proc in Dataset_lst:
                                timdist_dat_dict = reload_time_dist_data_comb_time(var,var_d[1]['mat'],var_grid,var_dim,var_d['d'],ldi,thd, time_datetime,time_d, ii,jj,iijj_ind,nz,ntime, grid_dict,z_meth,zz,zi,lon_d,lat_d,xarr_dict,do_mask_dict,load_second_files,Dataset_lst,configd,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time,secdataset_proc = secdataset_proc)

                                if figtd is not None:
                                    if plt.fignum_exists(figtd.number):
                                        plt.close(figtd)

                                
                                if var_dim[var] == 4:  
                                    td_title_str = 'Time-Distance %s (%s) for %s (through %s)'%(nice_varname_dict[var],nice_lev, dataset_lab_d[secdataset_proc],lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0])
                                
                                elif var_dim[var] == 3:
                                    td_title_str = 'Time-Distance %s for %s (through %s)'%(nice_varname_dict[var],dataset_lab_d[secdataset_proc],lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[0])
                                


                                #secdataset_proc
                                figtd = plt.figure()
                                figtd.set_figheight(10*1.2)
                                figtd.set_figwidth(8*1.5)
                                #figtd.suptitle('%s Time-Distance for %s'%(nice_varname_dict[var], dataset_lab_d[secdataset_proc]), fontsize = 20)
                                figtd.suptitle(td_title_str, fontsize=figsuptitfontsize)
                                plt.subplots_adjust(top=0.90,bottom=0.05,left=0.05,right=1,hspace=0.25,wspace=0.6)
                                axtd = [plt.subplot(211),plt.subplot(212)]
                                paxtd = []
                                #paxtd.append(axtd[0].pcolormesh(timdist_dat_dict['x']['t'], timdist_dat_dict['x']['x'],timdist_dat_dict['x'][secdataset_proc][:,0,:].T))
                                #paxtd.append(axtd[1].pcolormesh(timdist_dat_dict['y']['t'], timdist_dat_dict['y']['x'],timdist_dat_dict['y'][secdataset_proc][:,0,:].T))
                                '''
                                if var_dim[var] == 4:
                                    paxtd.append(axtd[0].pcolormesh(timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['data'][:,0,:].T))
                                    paxtd.append(axtd[1].pcolormesh(timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['data'][:,0,:].T))
                                elif var_dim[var] == 3:
                                    paxtd.append(axtd[0].pcolormesh(timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['data'][:,:].T))
                                    paxtd.append(axtd[1].pcolormesh(timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['data'][:,:].T))
                                '''
                                paxtd.append(axtd[0].pcolormesh(timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['x']['Sec Grid'][secdataset_proc]['data'][:,:].T))
                                paxtd.append(axtd[1].pcolormesh(timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['t'], timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['x'],timdist_dat_dict['y']['Sec Grid'][secdataset_proc]['data'][:,:].T))
                                plt.colorbar(paxtd[0], ax = axtd[0])
                                plt.colorbar(paxtd[1], ax = axtd[1])
                                axtd[0].set_ylim(cur_xlim)
                                axtd[1].set_ylim(cur_ylim)

                                set_perc_clim_pcolor_in_region(5,95,ax = axtd[0])
                                set_perc_clim_pcolor_in_region(5,95,ax = axtd[1])

                                figtd.show()


                                td_close_win_meth = 1
                                # 1: closes when you click on it
                                # 2: stays open till you close it

                                if td_close_win_meth == 1:
                                    xclickax = figtd.add_axes([0,0,1,1], frameon=False)
                                    xclickax.axis('off')

                                # redraw canvas
                                figtd.canvas.draw()
                                
                                #flush canvas
                                figtd.canvas.flush_events()
                                
                                # Show plot, and set it as the current figure and axis
                                figtd.show()
                                plt.figure(figtd.figure)
                                

                                if td_close_win_meth == 1:
                                    plt.sca(xclickax)
                                
                                    ###################################
                                    # Close on button press: #JT COBP #
                                    ###################################


                                    close_tdax = False
                                    while close_tdax == False:

                                        # get click location
                                        tmptdbutloc = plt.ginput(1, timeout = 3) #[(0.3078781362007169, 0.19398809523809524)]
                                                
                                        #pdb.set_trace()
                                        if len(tmptdbutloc)!=1:
                                            #print('tmptdbutloc len != 1',tmptdbutloc )
                                            #close_tdax = True
                                            continue
                                            #pdb.set_trace()
                                        else:
                                            if len(tmptdbutloc[0])!=2:
                                                close_tdax = True
                                                #print('tmptdbutloc[0] len != 2',tmptdbutloc )
                                                continue
                                                #pdb.set_trace()
                                            # was a button clicked?
                                            # if so, record which and allow the window to close
                                            if (tmptdbutloc[0][0] >= 0) & (tmptdbutloc[0][0] <= 1) & (tmptdbutloc[0][1] >= 0) & (tmptdbutloc[0][1] <= 1):
                                                #pdb.set_trace()
                                                close_tdax = True

                                        # quit of option box is closed without button press.
                                        if plt.fignum_exists(figtd) == False:
                                            close_tdax = True
                                            
                                    
                                    # close figure
                                    if close_tdax:
                                        if figtd is not None:
                                            if plt.fignum_exists(figtd.number):
                                                plt.close(figtd)



                            #except:
                            #    print('time-distance plot failed')
                            #    pdb.set_trace()
                    
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
                                    #### if fig zorder = 1
                                    '''
                                    tmpxsect_ax,tmpxsect_ii,tmpxsect_jj,tmpxsect_ti,tmpxsect_zz, tmpxsect_sel_xlocval,tmpxsect_sel_ylocval = indices_from_ginput_ax(ax,tmpxsectloc[0],tmpxsectloc[1], thd,ew_line_x = lon_d[1][jj,:],ew_line_y = lat_d[1][jj,:],ns_line_x = lon_d[1][:,ii],ns_line_y = lat_d[1][:,ii])
                                    pdb.set_trace()
                                    '''
                                    #### if fig zorder = -5
                                    tmpxsect_jj,tmpxsect_ii = ind_from_lon_lat('Dataset 1',configd,xypos_dict, lon_d,lat_d, thd,rot_dict,tmpxsectloc[0],tmpxsectloc[1])
                                    tmpxsect_ii = int(tmpxsect_ii)
                                    tmpxsect_jj = int(tmpxsect_jj)
                                    tmpxsect_ax = 0
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
                                if ((configd[1] == configd[2])|((configd[1].upper() in ['AMM15','C09P2','CO9P2']) & (configd[2].upper() in ['AMM15','C09P2','CO9P2']))):
                                
                                    figxs = plt.figure()
                                    figxs.set_figheight(10*1.2)
                                    figxs.set_figwidth(8*1.5)
                                    figxs.suptitle('Cross-section: %s'%nice_varname_dict[var], fontsize = 20)
                                    plt.subplots_adjust(top=0.90,bottom=0.05,left=0.05,right=1,hspace=0.25,wspace=0.6)
                                    axxs = [plt.subplot(311),plt.subplot(312),plt.subplot(313)]
                                    paxxs = []
                                    for xi,tmp_datstr in enumerate(Dataset_lst): paxxs.append(axxs[xi].pcolormesh(tmp_xsect_x[tmp_datstr],tmp_xsect_z[tmp_datstr],tmp_xsect_dat[tmp_datstr]))
                                    paxxs.append(axxs[2].pcolormesh(tmp_xsect_x[xsect_secdataset_proc],tmp_xsect_z[xsect_secdataset_proc],tmp_xsect_dat[Dataset_lst[1]] - tmp_xsect_dat[Dataset_lst[0]], cmap = matplotlib.cm.seismic))
                                    for tmpax  in axxs: tmpax.invert_yaxis()
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_title(dataset_lab_d[tmp_datstr])
                                    axxs[2].set_title('%s - %s'%(dataset_lab_d[Dataset_lst[1]],dataset_lab_d[Dataset_lst[0]]))
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
                                    for xi,tmp_datstr in enumerate(Dataset_lst): axxs[xi].set_title(dataset_lab_d[tmp_datstr])
                                    xs_ylim = np.array(axxs[0].get_ylim())
                                    xs_xlim = np.array(axxs[0].get_xlim())
                                    tmp_xsect_zmat = tmp_xsect_z[tmp_datstr][~(tmp_xsect_x[tmp_datstr]*tmp_xsect_z[tmp_datstr]*tmp_xsect_dat[tmp_datstr]).mask]
                                    if tmp_xsect_zmat.size>0:
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
                                    
                                    for xi,tmp_datstr in enumerate(Dataset_lst):axxs[0].plot(tmp_xsect_x[tmp_datstr],tmp_xsect_dat[tmp_datstr],label = dataset_lab_d[tmp_datstr])
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
                                        tmp_T_data[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj,ii].load())
                                        tmp_S_data[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,jj,ii].load())
                                        tmp_gdept[tmp_datstr] = np.array(grid_dict[tmp_datstr]['gdept'])[:,jj,ii]
                                        tmp_mld1[tmp_datstr] = np.ma.masked
                                        tmp_mld2[tmp_datstr] = np.ma.masked
                                        if 'mld25h_1' in var_d[th_d_ind]['mat']: tmp_mld1[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['mld25h_1'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][jj,ii].load())
                                        if 'mld25h_2' in var_d[th_d_ind]['mat']: tmp_mld2[tmp_datstr] = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['mld25h_2'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][jj,ii].load())

                                    else:
                                        if not np.ma.is_masked(iijj_ind[tmp_datstr]['jj']*iijj_ind[tmp_datstr]['ii']):
                                            if 'votemper' in var_d[th_d_ind]['mat']:tmp_T_data[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                            if 'vosaline' in var_d[th_d_ind]['mat']:tmp_S_data[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['vosaline'][ti,:,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                            if 'mld25h_1' in var_d[th_d_ind]['mat']:tmp_mld1[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['mld25h_1'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                            if 'mld25h_2' in var_d[th_d_ind]['mat']:tmp_mld2[tmp_datstr]  = np.ma.masked_invalid(xarr_dict[tmp_datstr][gr_1st][ldi].variables['mld25h_2'][ti,thd[th_d_ind]['y0']:thd[th_d_ind]['y1']:thd[th_d_ind]['dy'],thd[th_d_ind]['x0']:thd[th_d_ind]['x1']:thd[th_d_ind]['dx']][iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']].load())
                                            tmp_gdept[tmp_datstr] =  np.array(grid_dict[tmp_datstr]['gdept'])[:,iijj_ind[tmp_datstr]['jj'],iijj_ind[tmp_datstr]['ii']]               
                                        else:
                                            if 'votemper' in var_d[th_d_ind]['mat']:tmp_T_data[tmp_datstr]  = np.ma.zeros((xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'].shape[1]))*np.ma.masked
                                            if 'vosaline' in var_d[th_d_ind]['mat']:tmp_S_data[tmp_datstr]  = np.ma.zeros((xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'].shape[1]))*np.ma.masked
                                            if 'mld25h_1' in var_d[th_d_ind]['mat']:tmp_mld1[tmp_datstr]  = np.ma.zeros((1))*np.ma.masked
                                            if 'mld25h_2' in var_d[th_d_ind]['mat']:tmp_mld2[tmp_datstr]  = np.ma.zeros((1))*np.ma.masked
                                            tmp_gdept[tmp_datstr] =  np.ma.arange(xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'].shape[1]*1.)/xarr_dict[tmp_datstr][gr_1st][ldi].variables['votemper'].shape[1]
                                 
                                            #pdb.set_trace()
                                    
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
                                for dsi,tmp_datstr in enumerate(Dataset_lst):axtp.plot(np.ma.masked,color = 'k', label =  dataset_lab_d[tmp_datstr], linestyle = linestyle_str[dsi])
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
                                #for dsi,tmp_datstr in enumerate(Dataset_lst): figts_lab_str = figts_lab_str + '\n\n%s'%dataset_lab_d[tmp_datstr]
                                #plt.text(0.5, 0.1, figts_lab_str, fontsize=14, transform=figts.transFigure, ha = 'left', va = 'bottom')
                                plt.text(0.5, 0.9, figts_lab_str, fontsize=14, transform=figts.transFigure, ha = 'left', va = 'bottom')

                                
                                TSfig_out_name = '%s/output_TSDiag_%s_%s_%s'%(fig_dir,fig_lab,lon_lat_to_str(lon_d[1][jj,ii],lat_d[1][jj,ii])[3],time_datetime[ti].strftime('%Y%m%dT%H%MZ'))
                                
                                if not os.path.exists(fig_dir):
                                    os.makedirs(fig_dir)
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
                                    
                                    #fsct_hov_dat = reload_hov_data_comb(var,var_d[1]['mat'],var_grid['Dataset 1'],var_d['d'],fcst_ldi, thd,time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict, load_second_files,Dataset_lst,configd)
                                    #fsct_hov_dat = reload_hov_data_comb_time(var,var_d[1]['mat'],var_grid['Dataset 1'],var_d['d'],fcst_ldi, thd,time_datetime, ii,jj,iijj_ind,nz,ntime, grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict, load_second_files,Dataset_lst,configd)
                                    fsct_hov_dat = reload_hov_data_comb_time(var,var_d[1]['mat'],var_grid,var_dim,var_d['d'],fcst_ldi, thd,time_datetime, time_d,ii,jj,iijj_ind,nz,ntime, grid_dict,lon_d,lat_d,xarr_dict,do_mask_dict, load_second_files,Dataset_lst,configd)
                                    
                                    for tmp_datstr in Dataset_lst:
                                        #print('fsct_hov_dat_dict',fcst_ldi,tmp_datstr,fsct_hov_dat_dict[tmp_datstr][fcst_ldi].shape,fsct_hov_dat[tmp_datstr].shape)                                        
                                        fsct_hov_dat_dict[tmp_datstr][fcst_ldi] = fsct_hov_dat[tmp_datstr]

                                    fsct_hov_x[fcst_ldi] = fsct_hov_dat['x'] + timedelta(hours = ld_time_offset[fcst_ldi])
                
                                    #fsct_ts_dat = reload_ts_data_comb(var,var_dim,var_grid['Dataset 1'],ii,jj,iijj_ind,fcst_ldi,fsct_hov_dat,time_datetime,time_d,z_meth,zz,zi,xarr_dict,do_mask_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,load_second_files)
                                    fsct_ts_dat = reload_ts_data_comb_time(var,var_dim,var_grid,ii,jj,iijj_ind,fcst_ldi,fsct_hov_dat,time_datetime,time_d,z_meth,zz,zi,lon_d,lat_d,xarr_dict,do_mask_dict,grid_dict,thd,var_d[1]['mat'],var_d['d'],nz,ntime,configd,Dataset_lst,load_second_files,do_LBC = do_LBC, do_LBC_d = do_LBC_d,LBC_coord_d = LBC_coord_d, EOS_d=EOS_d,do_match_time=do_match_time)
                                    
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
                                axfc[0].set_title(dataset_lab_d['Dataset 1'])
                                if load_second_files:       
                                    axfc[1].plot(fsct_ts_x,fsct_ts_dat_dict['Dataset 2'][:,:], '0.5' )                   
                                    axfc[1].plot(fsct_ts_x[0,:],fsct_ts_dat_dict['Dataset 2'][0,:],'ro' )
                                    axfc[1].plot(fsct_ts_x[-1,:],fsct_ts_dat_dict['Dataset 2'][-1,:],'x', color = '0.5')
                                    axfc[1].set_title(dataset_lab_d['Dataset 2'])
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


                            elif mouse_info['button'].name == 'MIDDLE':

                                # Bring up a options window for Obs
                                
                                # button names                            
                                grad_but_names = ['grad_meth','grad_2d_meth','grad_abs_pre','grad_abs_post','grad_dx_d_dx']# 'grad_regrid_xy',
                                
                                # button switches  
                                grad_but_sw = {}
                                grad_but_sw['grad_abs_pre'] =  {'v':grad_abs_pre, 'T':'Pre-proc: |x|','F':'Pre-proc: x','T_col': '0.5','F_col':'k'}
                                grad_but_sw['grad_abs_post'] = {'v':grad_abs_post, 'T':'Post-proc: |x|','F':'Post-proc: x','T_col': '0.5','F_col':'k'}
                                #grad_but_sw['grad_regrid_xy'] = {'v':grad_regrid_xy, 'T':'Regrid xy coords','F':'Original xy coords','T_col': 'k','F_col':'0.5'}
                                grad_but_sw['grad_dx_d_dx'] =  {'v':grad_dx_d_dx, 'T':'dx(dy/dx)','F':'dy/dx', 'T_col': '0.5','F_col':'k'}
                                
                                grad_but_sw['grad_meth'] = {'v':grad_meth,0:'Grad Meth: Centred Diff',1: 'Grad Meth: Forward Diff'}
                                grad_but_sw['grad_2d_meth'] = {'v':grad_2d_meth,0:'Grad 2D Method: magnitude',1: 'Grad 2D Method: d/dx',2: '2D Method: d/dy'}
                                gradbut_sel = pop_up_opt_window(grad_but_names, opt_but_sw = grad_but_sw)

                                

                                # Set the main figure and axis to be current
                                plt.figure(fig.figure)
                                plt.sca(clickax)

                                if gradbut_sel ==  'grad_meth':
                                    grad_meth +=1
                                    if grad_meth ==2:grad_meth = 0
                                elif gradbut_sel ==  'grad_2d_meth':
                                    grad_2d_meth +=1
                                    if grad_2d_meth ==3:grad_2d_meth = 0
                                elif gradbut_sel ==  'grad_abs_pre':
                                    grad_abs_pre = not grad_abs_pre
                                elif gradbut_sel ==  'grad_abs_post':
                                    grad_abs_post = not grad_abs_post
                                elif grad_regrid_xy ==  'grad_regrid_xy':
                                    grad_regrid_xy = not grad_regrid_xy
                                elif gradbut_sel ==  'grad_dx_d_dx':
                                    grad_dx_d_dx = not grad_dx_d_dx

                                print('grad_meth:',grad_meth)
                                print('grad_abs_pre:',grad_abs_pre)
                                print('grad_abs_post:',grad_abs_post)
                                print('grad_regrid_xy:',grad_regrid_xy)
                                print('grad_dx_d_dx:',grad_dx_d_dx)

                                # if the button closed was one of the Obs types, add or remove from the hide list
                                #for m_var in MLD_var_lst:  
                                #    if mld_but_sw[m_var]['v']:
                                #        MLD_var = m_var
                                        
                        




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
                                Sec_regrid_slice = False
                                func_but_text_han['Sec Grid'].set_color('k')
                            else:
                                Sec_regrid = True
                                Sec_regrid_slice = True
                                func_but_text_han['Sec Grid'].set_color('darkgreen')
                                # when loading with Sec Grid, there is another field in the map_dat_dict
                                #   if its not there, reload.
                                # don't need to check other datasets, as if its in 2, itll be in all secondary datasets?
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
                            #pdb.set_trace()

                            tmp_secdataset_proc_button_name = func_but_text_han[but_name].get_text()
                            # if clicked on Dat1-Dat2: and its already Dat1-Dat2
                            #   Change value

                            secdataset_proc = but_name 
                            if but_name not in Dataset_lst:
                                #if already seleted
                                
                                tmpdataset_oper_char_ind = int(np.where(np.array([ss in tmpdataset_oper_lst for ss in tmp_secdataset_proc_button_name]))[0][0])
                                tmpdataset_oper_b = tmp_secdataset_proc_button_name[tmpdataset_oper_char_ind]
                                #pdb.set_trace()
                                tmpdataset_dat_1st = tmp_secdataset_proc_button_name[:tmpdataset_oper_char_ind]
                                tmpdataset_dat_2nd = tmp_secdataset_proc_button_name[tmpdataset_oper_char_ind+1:]
                                    
                                print('but_name, Dataset_lst',but_name, Dataset_lst)
                                tmp_secdataset_proc_button_name = func_but_text_han[but_name].get_text()
                                print('tmp_secdataset_proc_button_name',tmp_secdataset_proc_button_name)
                                print("mouse_info['button'].name",mouse_info['button'].name)
                                #pdb.set_trace()
                                if but_name == secdataset_proc:
                                #if secdataset_proc == tmp_secdataset_proc_button_name:
                                #if but_name == tmp_secdataset_proc_button_name:
                                    #find the dataset numbers, and operation type

                                    #tmpdataset_oper_char_ind = int(np.where(np.array([ss in tmpdataset_oper_lst for ss in tmp_secdataset_proc_button_name]))[0][0])
                                    #tmpdataset_oper_b = tmp_secdataset_proc_button_name[tmpdataset_oper_char_ind]
                                    #tmpdataset_dat_1st = tmp_secdataset_proc_button_name[:tmpdataset_oper_b]
                                    #tmpdataset_dat_2nd = tmp_secdataset_proc_button_name[tmpdataset_oper_b+1:]
                                    '''
                                    if len(tmp_secdataset_proc_button_name) == 5:
                                        tmpdataset_oper_b = tmp_secdataset_proc_button_name[2]
                                        tmpdataset_dat_1st =tmp_secdataset_proc_button_name[:2]
                                        tmpdataset_dat_2nd =tmp_secdataset_proc_button_name[3:]
                                    elif len(tmp_secdataset_proc_button_name) == 9:
                                        tmpdataset_oper_b = tmp_secdataset_proc_button_name[4]
                                        tmpdataset_dat_1st =tmp_secdataset_proc_button_name[:4]
                                        tmpdataset_dat_2nd =tmp_secdataset_proc_button_name[5:]
                                    '''
                                    # cycle through next or previous cycle of operation, and update text
                                    if tmpdataset_oper in tmpdataset_oper_lst:    
                                        if mouse_info['button'].name == 'LEFT':
                                            tmpdataset_oper_a=tmpdataset_oper_lst[int((np.where(tmpdataset_oper == np.array(tmpdataset_oper_lst))[0]+1)%3)]
                                            print('left press,was, now,',tmpdataset_oper_b, tmpdataset_oper_a)

                                            secdataset_proc = tmpdataset_oper_a.join(but_name.split(tmpdataset_oper))

                                        elif mouse_info['button'].name == 'RIGHT':
                                            tmpdataset_oper_a=tmpdataset_oper_lst[int((np.where(tmpdataset_oper == np.array(tmpdataset_oper_lst))[0]-1)%3)]
                                            print('right press,was, now,',tmpdataset_oper_b, tmpdataset_oper_a)
                                            
                                            secdataset_proc = tmpdataset_oper_a.join(but_name.split(tmpdataset_oper))

                                        else:
                                            tmpdataset_oper_a = tmpdataset_oper_b
                                            tmpdataset_dat_1st,tmpdataset_dat_2nd = tmpdataset_dat_2nd,tmpdataset_dat_1st
                                            print('middle  press,was, now,',tmp_secdataset_proc_button_name, tmpdataset_dat_1st  + tmpdataset_oper_a + tmpdataset_dat_2nd)

                                            secdataset_proc = tmpdataset_oper.join(but_name.split(tmpdataset_oper)[::-1])

                                            #pdb.set_trace()
                                        func_but_text_han[but_name].set_text(tmpdataset_dat_1st  + tmpdataset_oper_a + tmpdataset_dat_2nd)

                                #pdb.set_trace()
                                #secdataset_proc = func_but_text_han[but_name].get_text()

                            #else:
                            #    secdataset_proc = but_name 

                            #obs_load_sel[secdataset_proc] = False

                            for tmpsecdataset_proc in secdataset_proc_list: func_but_text_han[tmpsecdataset_proc].set_color('k')


                            func_but_text_han[but_name].set_color('darkgreen')

                            if do_ensemble:
                                Ens_stat = None
                                for ens_stat in ens_stat_lst: func_but_text_han[ens_stat].set_color('k')

                            

                            fig_tit_str_lab = ''
                            #if dataset_lab_d['Dataset 1'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 1 = %s;'%dataset_lab_d['Dataset 1']
                            #if dataset_lab_d['Dataset 2'] is not None: fig_tit_str_lab = fig_tit_str_lab + ' Dataset 2 = %s;'%dataset_lab_d['Dataset 2']
                            for tmp_datstr in Dataset_lst:
                                #if dataset_lab_d[tmp_datstr] is not None: 
                                fig_tit_str_lab = fig_tit_str_lab + ' %s = %s;'%(tmp_datstr,dataset_lab_d[tmp_datstr])


                            cur_fig_tit_str_lab = ''
                            if load_second_files == False:
                                cur_fig_tit_str_lab = dataset_lab_d['Dataset 1']
                            else:
                                if secdataset_proc in Dataset_lst:
                                    cur_fig_tit_str_lab = '%s'%dataset_lab_d[secdataset_proc]
                                else:
                                    #pdb.set_trace()
                                    tmpdataset_1 = 'Dataset ' + secdataset_proc[3]
                                    tmpdataset_2 = 'Dataset ' + secdataset_proc[8]
                                    tmpdataset_oper = secdataset_proc[4]
                                    '''
                                    tmpdataset_oper_char_ind = int(np.where(np.array([ss in tmpdataset_oper_lst for ss in tmp_secdataset_proc_button_name]))[0][0])
                                    
                                    tmpdataset_1 = 'Dataset ' + secdataset_proc[tmpdataset_oper_char_ind-1]
                                    tmpdataset_2 = 'Dataset ' + secdataset_proc[-1]
                                    '''
                                    tmpdataset_oper = secdataset_proc[tmpdataset_oper_char_ind]
                                    if tmpdataset_oper == '-':
                                        cur_fig_tit_str_lab = '%s minus %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
                            
                                    elif tmpdataset_oper == '/':
                                        cur_fig_tit_str_lab = '%s over %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
                            
                                    elif tmpdataset_oper == '%':
                                        cur_fig_tit_str_lab = '%s percent diff %s'%(dataset_lab_d[tmpdataset_1],dataset_lab_d[tmpdataset_2])
                            
                            
                            fig_tit_str_lab = fig_tit_str_lab + ' Showing %s.'%(cur_fig_tit_str_lab)
                            
                            fig.suptitle(fig_tit_str_int + '\n' + fig_tit_str_lab, fontsize=figsuptitfontsize)
                            

 

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
                            #zm_2d_meth_lst = ['zm','zx','zn','zs','zd']
                            #zm_2d_meth_full_lst = ['Depth-Mean','Depth-Max','Depth-Min','Depth Spike Mag','Depth-Std']
                            
                            if var_dim[var] == 4:

                                if but_name == 'Depth-Mean':
                                    if z_meth in zm_2d_meth_lst:
                                        # switch button
                                        if mouse_info['button'].name == 'LEFT':
                                            DepthMean_sw+=1
                                        else:
                                        #elif mouse_info['button'].name == 'RIGHT':
                                            DepthMean_sw-=1

                                    #if DepthMean_sw == 4: DepthMean_sw = 0
                                    #if DepthMean_sw == -1: DepthMean_sw = 3

                                    if DepthMean_sw == len(zm_2d_meth_lst): DepthMean_sw = 0
                                    if DepthMean_sw == -1: DepthMean_sw = len(zm_2d_meth_lst)-1
                                    
                                    func_but_text_han['Depth-Mean'].set_text(zm_2d_meth_full_lst[DepthMean_sw])
                                    z_meth = zm_2d_meth_lst[DepthMean_sw]


                                
                                if but_name == 'Surface':z_meth = 'ss'
                                if but_name == 'Near-Bed': z_meth = 'nb'
                                if but_name == 'Surface-Bed': z_meth = 'df'
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
            #print("selected ii = %i,jj = %i,ti = %i,zz = %i, var = '%s'"%(ii,jj, ti, zz,var))
            print("selected --ii %i --jj %i --ti  %i --zz %i --var '%s'"%(ii,jj, ti, zz,var))
            # after selected indices and vareiabels, delete plots, ready for next cycle
            #pdb.set_trace()
            for tmp_cax in cax:tmp_cax.remove()
            
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

            '''
            cs_plot_1_pop = cs_plot_1.pop()
            cs_plot_1_pop.remove()
            cs_plot_2_pop = cs_plot_2.pop()
            cs_plot_2_pop.remove()

            if Sec_regrid_slice:
                if secdataset_proc in Dataset_lst[1:]:                    
                    csrg_plot_1_pop = csrg_plot_1.pop()
                    csrg_plot_1_pop.remove()
                    csrg_plot_2_pop = csrg_plot_2.pop()
                    csrg_plot_2_pop.remove()
            '''

            
            for tmpcrshr_ax in crshr_ax:
                rem_loc = tmpcrshr_ax.pop(0)
                rem_loc.remove()
            
            # sometime when it crashes, it adds additional colorbars. WE can catch this be removing any colorbars from the figure... 
            #   however, this doesn't reset the axes size, so when the new colorbar is added, the axes is reduced in size. 
            #   maybe better to specify axes and colorbar location, rathar than using subplot, and colorbar().
            for child in fig.get_children():
                child.__class__.__name__
                if child.get_label() == '<colorbar>': child.remove()
            
            if verbose_debugging: print('Cycle', datetime.now())

            if do_timer: timer_lst.append(('Cycle ended',datetime.now()))
            if do_memory & do_timer: timer_lst[-1]= timer_lst[-1] + (psutil.Process(os.getpid()).memory_info().rss/1024/1024,)


def main():

    # Load help text
    nemo_slice_zlev_helptext = load_nemo_slice_zlev_helptext()

    if sys.argv.__len__() > 1:

        # Load argparse parser, setting all the command line keyword arguments
        parser = load_NEMO_nc_viewer_parser(nemo_slice_zlev_helptext)

        # Load read command line keyword arguments 
        args = parser.parse_args()



        ############################################################
        # Handling of Bool variable types
        #
        # manage the handling of the boolean argpass options, and fill in a dictionary of their values
        # separating those that default to True
        argparse_bool_T = ['clim_pair','fig_cutout','do_match_time','trim_files','Obs_pair_loc','Obs_AbsAnom']
        # from those that default to False
        argparse_bool_F = ['allow_diff_time','clim_sym','hov_time','justplot','use_cmocean','verbose_debugging','do_timer','do_memory','do_ensemble','do_mask','do_addtimedim','do_all_WW3','do_cont','trim_extra_files','Obs_hide','use_xarray_gdept','Obs_hide_edges','Obs_AbsAnom','Time_Diff','Obs_pair_loc','Obs_show_with_diff_var']
        


        argparse_bool_dict = process_argparse_bool(args,argparse_bool_T,argparse_bool_F)
        ############################################################


        # Assume the first dataset (in the positional argument) is on the T grid.
        # You can specify what it is (with --gr_1st U).

        if (args.gr_1st) is None:
            gr_1st = 'T'

            # if first config is and LBC, use T_1, rather than T
            if args.config[-3:].upper() == 'LBC':
                gr_1st = 'T_1'
        else:
            gr_1st = args.gr_1st


        # Depreciated options
        z_meth_in = None   
        preload_data_in=True

        
        
        print(args.fname_lst)

        # setting up filename dictionary
        fname_dict, load_second_files = process_argparse_fname_dict(args,gr_1st)

        dataset_lst = [ ss for ss in fname_dict.keys() ] 
        nDataset = len(dataset_lst)

        # setting up configs dictionary
        configd = process_argparse_configd(args,dataset_lst)


        configlst = np.array([configd[ss] for ss in (configd)])
        uniqconfig = np.unique(configlst)


        # setting up dataset label dictionary
        dataset_lab_d = process_argparse_dataset_lab_d(args)

        # setting up Thinning dictionary
        thd = process_argparse_thd(args,configd, dataset_lst, nDataset)
       
        # print out thd
        for dsi in range(1,nDataset+1): print(thd[dsi])

        # setting up Observations dictionary
        Obs_dict_in = process_argparse_Obs_dict(args)
 
        # When comparing files/models, only variables that are common to both Datasets are shown. 
        # If comparing models with different names for the same variables, they won't be shown, 
        # as temperature and votemper will be considered different.

        # We can use xarray to rename the variables as they are loaded to overcome this, using a rename_dictionary.
        # i.e. rename any variables called tmperature or temp to votemper etc.
        # to do this, we use the following command line arguments:
        # --rename_var votemper temperature temp --rename_var vosaline salinity sal 
        # where each variable has its own instance, and the first entry is what it will be renamed too, 
        # and the remaining entries are renamed. 
        xarr_rename_master_dict = process_argparse_rename_var(args)

    
        # If files have more than grid, with differing dimension for each, you can enforce the dimenson for each grid.
        # For example, the SMHI BAL-MFC NRT system (BALMFCorig) hourly surface files hvae the T, U, V and T_inner grid in the same file. 
        # Load the smae file in for each grid:
      
        force_dim_d_in = process_argparse_forced_dim(args, dataset_lst, nDataset)

        # set up empty EOS dictionary
        EOS_d = process_argparse_EOS(args, dataset_lst)
        Obs_Type_load_dict = process_argparse_Obs_type_hide(args,argparse_bool_dict)


        define_time_dict = process_argparse_define_time(args, dataset_lst, fname_dict)

        #pdb.set_trace()
        nemo_slice_zlev(zlim_max = args.zlim_max,
            dataset_lab_d = dataset_lab_d,configd = configd,thd = thd,fname_dict = fname_dict,
            load_second_files = load_second_files,
            clim_sym = argparse_bool_dict['clim_sym'], clim = args.clim, clim_pair = argparse_bool_dict['clim_pair'],hov_time = argparse_bool_dict['hov_time'],
            allow_diff_time = argparse_bool_dict['allow_diff_time'],preload_data = preload_data_in,
            do_grad = args.do_grad,do_cont = argparse_bool_dict['do_cont'],trim_extra_files = argparse_bool_dict['trim_extra_files'],trim_files = argparse_bool_dict['trim_files'],
            use_cmocean = argparse_bool_dict['use_cmocean'], date_fmt = args.date_fmt,
            Time_Diff = argparse_bool_dict['Time_Diff'],
            define_time_dict = define_time_dict,
            fig_fname_lab = args.fig_fname_lab,
            justplot = argparse_bool_dict['justplot'],justplot_date_ind = args.justplot_date_ind,
            justplot_secdataset_proc = args.justplot_secdataset_proc,
            justplot_z_meth_zz = args.justplot_z_meth_zz,
            ii = args.ii, jj = args.jj, ti = args.ti, zz = args.zz, 
            lon_in = args.lon, lat_in = args.lat, date_in_ind = args.date_ind,
            var = args.var, z_meth = z_meth_in,
            vis_curr = args.vis_curr,
            xlim = args.xlim,ylim = args.ylim,
            secdataset_proc = args.secdataset_proc,
            ld_lst = args.ld_lst, ld_lab_lst = args.ld_lab_lst, ld_nctvar = args.ld_nctvar,
            resample_freq = args.resample_freq,
            Obs_dict = Obs_dict_in,Obs_hide = argparse_bool_dict['Obs_hide'],
            Obs_AbsAnom = argparse_bool_dict['Obs_AbsAnom'], Obs_hide_edges = argparse_bool_dict['Obs_hide_edges'],
            Obs_pair_loc = argparse_bool_dict['Obs_pair_loc'], Obs_anom_clim = args.Obs_anom_clim,
            Obs_Type_load_dict = Obs_Type_load_dict,Obs_show_with_diff_var = argparse_bool_dict['Obs_show_with_diff_var'],
            fig_dir = args.fig_dir, fig_lab = args.fig_lab,fig_cutout = argparse_bool_dict['fig_cutout'],
            verbose_debugging = argparse_bool_dict['verbose_debugging'],do_timer = argparse_bool_dict['do_mask'],do_memory = argparse_bool_dict['do_memory'],
            do_ensemble = argparse_bool_dict['do_ensemble'],do_mask = argparse_bool_dict['do_mask'],
            use_xarray_gdept = argparse_bool_dict['use_xarray_gdept'],
            force_dim_d = force_dim_d_in,xarr_rename_master_dict=xarr_rename_master_dict,EOS_d = EOS_d,gr_1st = gr_1st,
            do_match_time = argparse_bool_dict['do_match_time'],do_addtimedim = argparse_bool_dict['do_addtimedim'], do_all_WW3=argparse_bool_dict['do_all_WW3'])


        exit()

    
if __name__ == "__main__":
    main()
