NEMO_nc_slevel_viewer readme
=========================================
(Jonathan Tinker, Met Office, 23/10/2024)
=========================================

Thanks you for downloading NEMO_nc_slevel_viewer!


The main code is in the NEMO_nc_slevel_viewer.py, with NEMO_nc_slevel_viewer_dev.py being a development branch 
(likely to be the same on github). 
NEMO_nc_slevel_viewer_lib.py is a library, containing most of the functions.

When running at the command line, at the very least, you need to give the configuration your using 
(AMM7, AMM15, CO9p2, ORCA12, ORCA025) and some NEMO NC files. 

The configuration name pulls in the correct configuration csv file (e.g. NEMO_nc_slevel_viewer_config_AMM15.csv), 
which contains the (local) location of the model mesh file, the name of the gdept variable within the mesh, and 
the default z sliceing methodology (effectively whether model z or s levels). There are also regridding files for 
the AMM configs (these are in the github repo).

The configuration files will have to be changed to point to your local copy the mesh file.

The code does not (currently?) cope with mesh files with land suppression.

if passing any wildcards, embed in ""

You (currently) always need to provide the T grid files, but then can add other grids.

Example useage:
====================

Simple case
-------------

python NEMO_nc_slevel_viewer.py amm7  /path/to/file.nc


python NEMO_nc_slevel_viewer.py amm7  "/path/to/files*.nc"
  
flist_ammT=$(echo "/path/to/files*.nc")
python NEMO_nc_slevel_viewer.py amm7  "$flist_ammT" 


Advanced case
-------------

flist_amm15_T_1=$(echo "/path/to/amm15/files/amm15.grid_T.nc")
flist_amm15_U_1=$(echo "/path/to/amm15/files/amm15.grid_U.nc")
flist_amm15_V_1=$(echo "/path/to/amm15/files/amm15.grid_V.nc")
flist_amm7_T_1=$(echo "/path/to/amm7/files/amm7vx.grid_T.nc")
flist_amm7_U_1=$(echo "/path/to/amm7/files/amm7vx.grid_U.nc")
flist_amm7_V_1=$(echo "/path/to/amm7/files/amm7vx.grid_V.nc")


fig_fname_lab_amm15=OS44_amm15
fig_fname_lab_amm7=OS44_amm7



##AMM7 and BGC (ERSEM)
python /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer/NEMO_nc_slevel_viewer_dev.py amm7  "$flist_amm7_T_1"    --U_fname_lst "$flist_amm7_U_1" --V_fname_lst "$flist_amm7_V_1"  --fig_fname_lab $fig_fname_lab_amm7  
  

##AMM15 
python /home/h01/hadjt/workspace/python3/NEMO_nc_slevel_viewer/NEMO_nc_slevel_viewer_dev.py amm15  "$flist_amm15_T_1"    --U_fname_lst "$flist_amm15_U_1" --V_fname_lst "$flist_amm15_V_1"  --fig_fname_lab $fig_fname_lab_amm15 --thin 5