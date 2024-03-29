
import numpy as np
from netCDF4 import Dataset

import sys

"""
slurm = True
#slurm = False
if sys.stdin.isatty():slurm = False


if slurm == False:


    import matplotlib.gridspec as gridspec
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import iris.plot as iplt
    import iris.quickplot as qplt
    import matplotlib.pyplot as plt
    #from rotate_wind_vectors import *
else:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

"""
import matplotlib.pyplot as plt

#from python3_plotting_function import set_perc_clim_pcolor, get_clim_pcolor, set_clim_pcolor,set_perc_clim_pcolor_in_region



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
    nz,nlon,nlat = gdept.shape

    gdept_ma = gdept
    gdept_ma_min = gdept_ma.min(axis = 0)
    gdept_ma_max = gdept_ma.max(axis = 0)
    gdept_ma_ptp = gdept_ma.ptp(axis = 0)
    #zind_mat = np.zeros(gdept.shape, dtype = 'int')
    xind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
    yind_mat = np.zeros(gdept.shape[1:], dtype = 'int')
    #for zi in range(nz): zind_mat[zi,:,:] = zi
    for xi in range(nlon): xind_mat[xi,:] = xi
    for yi in range(nlat): yind_mat[:,yi] = yi

    ind1 = (gdept_ma<z_lev).sum(axis = 0).data.astype('int')
    ind1[ind1 == nz] = 0
    ind2 = (nz-1)-(gdept_ma>z_lev).sum(axis = 0).data.astype('int')
    #tmpind2 = (gdept_ma>z_lev).sum(axis = 0).data.astype('int')
    #ind2 = (nz-1)-tmpind2

    #plt.pcolormesh(gdept_ma[ind1,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()
    #plt.pcolormesh(gdept_ma[ind2,xind_mat,yind_mat]) ; plt.colorbar() ; plt.show()

    z_ind1 = gdept_ma[ind1,xind_mat,yind_mat]
    z_ind2 = gdept_ma[ind2,xind_mat,yind_mat]
    dz_ind = z_ind1-z_ind2

    zdist1 = z_ind1 - z_lev
    zdist2 = z_lev - z_ind2

    zdist1_norm = zdist1/dz_ind
    zdist2_norm = zdist2/dz_ind

    wgt1 = zdist2_norm
    wgt2 = zdist1_norm


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

def nearbed_index(filename, variable_4d,nemo_nb_i_filename = '/home/h01/hadjt/Work/Programming/Scripts/reffiles/nemo_nb_i.nc'):


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

    #nemo_nb_i_filename = '/home/cr/ocean/hadjt/data/reffiles/nemo_nb_i.nc'
    #nemo_nb_i_filename = '/home/h01/hadjt/Work/Programming/Scripts/reffiles/nemo_nb_i.nc'
    rootgrp_out = Dataset(nemo_nb_i_filename, 'w', format='NETCDF3_CLASSIC')
    #rootgrp_out.createDimension('x',297)
    #rootgrp_out.createDimension('y',375)
    #rootgrp_out.createDimension('z',51)
    rootgrp_out.createDimension('x',ni)
    rootgrp_out.createDimension('y',nj)
    rootgrp_out.createDimension('z',nz)
    nb_i_out = rootgrp_out.createVariable('nb_i','i4',('z','y','x',),fill_value = -99)
    tmask_out = rootgrp_out.createVariable('t_mask','i4',('z','y','x',),fill_value = -99)
    nb_i_out[:,:] = nbind
    tmask_out[:,:] = tmask
    rootgrp_out.close()


    #nemo_nb_i_filename = '/home/h01/hadjt/Work/Programming/Scripts/reffiles/nemo_nb_i.nc'
    rootgrp_in = Dataset(nemo_nb_i_filename, 'r', format='NETCDF3_CLASSIC')
    nb_i_in = (rootgrp_in.variables['nb_i'][:,:,:] == 1)
    tmask = (rootgrp_in.variables['t_mask'][:,:,:] == 1)

    return nbind,tmask


def load_nearbed_index(nemo_nb_i_filename = '/home/h01/hadjt/Work/Programming/Scripts/reffiles/nemo_nb_i.nc'):


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



if __name__ == "__main__":
    main()
