import pandas as pd
import arcpy
from arcpy.ia import Foreach
import numpy as np
from scipy.stats import linregress
from itertools import chain
import os
import logging
import traceback
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ===============================================================
# Step 0: Settings
# ===============================================================
#Main directory that contains GDB and results  
working_directory = r"D:\Projects\RSA_py_codes\3_LSI_py"
#Output directory that contains raster results
working_directory_output = os.path.join(working_directory, "LSI_Output")
results_directory = os.path.join(working_directory_output, "results")
#File Geodatabase that contains all raster input layers
fgdb_name = "LSI_Inputs.gdb"
# Feature layer that contains points
feature_point = "Training_Samples"

# Feature layer that contains points for LSI Normalization
feature_point_LSI_norm = "Validation_Samples"

#default fields in feature point
def_fields = ['OBJECTID','Shape','id','LS1NOLS0','Class','Slp_prc','Category']
#Raster input layer those organized as group. Layer names SHOULD BE unique and
#and DO NOT EXCEED 30 chars in the name
raster_groups = [
    ["API",
     "Annual_Precipitation",
     "Daily_Precipitation",
     "Annual_Temperature",
     "Daily_Temperature"],

    ["Elevation",
     "Max_Elevation4x4",
     "Slope",
     "Max_Slope4x4",
     "TRI",
     "TWI",
     "Aspect",
     "Curvature",
     "Drainage_Density",
     "Flow_Accumulation",
     "Slope_Length",
     "Plan_Curvature",
     "Profile_Curvature",
     "SPI",
     "TPI"],

    ["Lithology",
     "Soil_Texture",
     "Soil_Erodibility",
     "Soil_Depth"],

    ["Land_Cover",
     "NDVI",
     "NDWI",
     "TNDVI"],

    ["Distance_to_Faults"],

    ["Distance_to_Roads"],

    ["Distance_to_Rivers"]
]

# Output names for grouped rasters
group_names = [
    "rr_grp0_clim",
    "rr_grp1_topo",
    "rr_grp2_soil",
    "rr_grp3_landcover",
    "rr_grp4_fault",
    "rr_grp5_road",
    "rr_grp6_river"
]

#Maximum p value for a layer that will not excluded in group
maxPValue = 0.001

# ===============================================================
# Step 1: Initialization
# ===============================================================
os.makedirs(working_directory_output, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)    
    
# -------------------------------------------------------------------
# Calculate Variance Inflation Factor (VIF)
# without statsmodels
# -------------------------------------------------------------------

def calculate_vif(df):

    vif_values = []

    for i in range(df.shape[1]):

        y = df.iloc[:, i]

        X_temp = df.drop(
            df.columns[i],
            axis=1
        )

        # Add intercept
        X_temp = np.column_stack(
            [np.ones(len(X_temp)), X_temp]
        )

        coefficients, residuals, _, _ = np.linalg.lstsq(
            X_temp,
            y,
            rcond=None
        )

        ss_total = np.sum(
            (y - np.mean(y)) ** 2
        )

        # Perfect correlation case
        if ss_total == 0 or len(residuals) == 0:

            vif = np.inf

        else:

            ss_residual = residuals[0]

            r2 = 1 - (
                ss_residual / ss_total
            )

            if r2 >= 1:

                vif = np.inf

            else:

                vif = 1 / (1 - r2)


        vif_values.append(vif)


    return vif_values
    
# ================================================================
# Full VIF Matrix
# ================================================================

def calculate_vif_matrix(df):

    vif_matrix = pd.DataFrame(
        np.zeros(
            (df.shape[1], df.shape[1])
        ),
        columns=df.columns,
        index=df.columns
    )


    for factor in df.columns:

        y = df[factor]

        X = df.drop(
            columns=[factor]
        )


        # Add intercept
        X = np.column_stack(
            [np.ones(len(X)), X]
        )


        coefficients, residuals, _, _ = np.linalg.lstsq(
            X,
            y,
            rcond=None
        )


        ss_total = np.sum(
            (y - np.mean(y)) ** 2
        )


        if ss_total == 0 or len(residuals) == 0:

            vif = np.inf

        else:

            ss_residual = residuals[0]

            r2 = 1 - (
                ss_residual / ss_total
            )


            if r2 >= 1:

                vif = np.inf

            else:

                vif = 1 / (1-r2)


        # Put VIF only on diagonal
        vif_matrix.loc[factor, factor] = vif


    return vif_matrix

# -------------------------------------------------------------------
# Invert raster
# -------------------------------------------------------------------

def invert_raster(raster_name):

    raster = arcpy.sa.Float(arcpy.sa.Raster(raster_name))

    raster_inverted = 1.0 - raster

    folder = working_directory_output

    name = os.path.splitext(
        os.path.basename(raster_name)
    )[0]

    name = name.replace("_Norm_uninv", "_Norm_inv")

    output_name = os.path.join(
        folder,
        name + ".tif"
    )

    if arcpy.Exists(output_name):
        arcpy.management.Delete(output_name)

    raster_inverted.save(output_name)

    logging.info(
        "Raster inversion completed: {}".format(output_name)
    )

    return output_name

# -------------------------------------------------------------------
# Delete unnecessary fields from feature class
# -------------------------------------------------------------------

def deleteUnnecessaryFields(inFeature, xfields):

    toBeDeletedFields = []

    allFields = arcpy.ListFields(inFeature)

    for f in allFields:

        if f.name not in xfields:
            toBeDeletedFields.append(f.name)


    if len(toBeDeletedFields) > 0:

        arcpy.management.DeleteField(
            inFeature,
            toBeDeletedFields
        )

        logging.info(
            "Deleted unnecessary fields: {}".format(
                toBeDeletedFields
            )
        )

    else:

        logging.info(
            "No unnecessary fields found."
        )
        
# -------------------------------------------------------------------
# Output folder for multicollinearity assessment
# -------------------------------------------------------------------
multicollinearity_directory = results_directory

os.makedirs(multicollinearity_directory, exist_ok=True)
       
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
    os.path.join(results_directory, "debug.log"),
    mode='w'
),
        logging.StreamHandler()
    ]
)

class LicenseError(Exception):
    pass

try:
    #Checking out Spatial Analyst license for Concurrent License environmet
    if arcpy.CheckExtension("Spatial") == "Available":
        logging.debug("Spatial license is Available")  
        arcpy.CheckOutExtension("Spatial")
        logging.debug("Checked Out Extension!") 
    else:
        # raise a custom exception
        logging.error("ERROR: Spatial license is NOT Available!!!")
        raise LicenseError

    working_full_directory = working_directory + "\\" + fgdb_name
    arcpy.env.workspace = working_full_directory
    arcpy.env.overwriteOutput = True

# Build full paths after workspace has been created
    feature_point = os.path.join(
    working_full_directory,
    feature_point
)

    feature_point_LSI_norm = os.path.join(
    working_full_directory,
    feature_point_LSI_norm
)

    raster_layers = list(chain.from_iterable(raster_groups))
    

    # ===============================================================
    # Step 2: Normalize all input factors to the range 0–1
    # ===============================================================

    logging.info("--------------------------------")
    logging.info("Step 2: Normalizing all input factors")
    logging.info("--------------------------------")

    normalized_raster_layers = []

    for raster_name in raster_layers:

        logging.info("Normalizing {}".format(raster_name))

        raster = arcpy.sa.Float(arcpy.sa.Raster(raster_name))

        rmin = float(
            arcpy.management.GetRasterProperties(
                raster_name,
                "MINIMUM"
            ).getOutput(0)
        )

        rmax = float(
            arcpy.management.GetRasterProperties(
                raster_name,
                "MAXIMUM"
            ).getOutput(0)
        )

        logging.info(
            "{} : Min = {:.6f}, Max = {:.6f}".format(
                raster_name,
                rmin,
                rmax
            )
        )

        if rmax == rmin:
            raise Exception(
                "Raster '{}' contains constant values and cannot be normalized.".format(
                    raster_name
                )
            )

        # ----------------------------------------------------------
        # Min-Max Normalization
        # Normalized = (X - Min) / (Max - Min)
        # ----------------------------------------------------------

        raster_norm = (raster - rmin) / (rmax - rmin)

        # Save normalized raster as GeoTIFF in the Output folder
        normalized_name = os.path.join(
            working_directory_output,
            raster_name + "_Norm_uninv.tif"
        )

        # Delete old raster if it already exists
        if arcpy.Exists(normalized_name):
            arcpy.management.Delete(normalized_name)

        # Save normalized raster
        raster_norm.save(normalized_name)

        # Store the full path for later processing
        normalized_raster_layers.append(normalized_name)

        logging.info("{} successfully normalized.".format(normalized_name))

    logging.info("--------------------------------")
    logging.info("All input factors successfully normalized.")
    logging.info("--------------------------------")

    # -------------------------------------------------------------------
    # Replace original raster names with normalized raster names
    # -------------------------------------------------------------------

    index = 0

    for group_index in range(len(raster_groups)):
        for raster_index in range(len(raster_groups[group_index])):

            raster_groups[group_index][raster_index] = (
                normalized_raster_layers[index]
            )

            index += 1

    raster_layers = normalized_raster_layers

    # -------------------------------------------------------------------
    # Build extraction fields using REAL normalized raster names
    # -------------------------------------------------------------------

    raster_field_names = []
    all_raster_extract_field_names = []

    for raster_path in normalized_raster_layers:

        # Example:
        # Annual_Precipitation_Norm.tif
        # becomes:
        # Annual_Precipitation_Norm

        raster_base = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        field_name = raster_base

        raster_field_names.append(field_name)

        all_raster_extract_field_names.append(
            [raster_path, field_name]
        )


    logging.info("Normalized rasters used for extraction:")
    for item in all_raster_extract_field_names:
        logging.info("{} -> {}".format(item[0], item[1]))

    #Delete old columns in feature layer
    
    print("--------------------------------")
    print("Workspace:", arcpy.env.workspace)
    print("Workspace exists:", arcpy.Exists(arcpy.env.workspace))
    print("Feature exists:", arcpy.Exists(feature_point))
    print("Full path exists:",
      arcpy.Exists(os.path.join(arcpy.env.workspace, feature_point)))
    print("--------------------------------")

    print("Feature classes found:")
    for fc in arcpy.ListFeatureClasses():
       print(fc)
    
    deleteUnnecessaryFields(feature_point, def_fields)
    
    arcpy.sa.ExtractMultiValuesToPoints(
        in_point_features=feature_point,
        in_rasters=all_raster_extract_field_names,
        bilinear_interpolate_values="NONE"
    )

    # =================================================================================================================
    # Step 3: Evaluation and Selection of Significant Factors Using Pearson Correlation Coefficients (r) and P-values
    # =================================================================================================================
    logging.debug("Step 3: Start Correlation")
    
    array_y = []
    with arcpy.da.SearchCursor(feature_point,["LS1NOLS0"]) as cur:
       for row in cur:
         array_y.append(row[0])
    ray = arcpy.da.TableToNumPyArray(feature_point, raster_field_names)

    resultCorrelationNp = np.zeros(len(raster_layers), dtype=[('Features', 'U256'), ('corr', 'float'), ('sign', 'float')])
    
    isInvertedLayer = False
    for index, item in enumerate(raster_layers):
        x = np.array(ray[raster_field_names[index]])
        y = np.array(array_y)
        res = linregress(y, x)
        resultCorrelationNp[index]['Features'] = raster_field_names[index]
        resultCorrelationNp[index]['corr'] = res.rvalue
        resultCorrelationNp[index]['sign'] = res.pvalue

        # DELETE
    # if(item == "Altitude_Clip_2"):
        # resultCorrelationNp[index]['corr'] = -0.1234
        # resultCorrelationNp[index]['sign'] = 0.00123
    
    #Checking for inverting
        if(resultCorrelationNp[index]['corr'] < 0):
            isInvertedLayer = True
            logging.debug("inverting raster "+str(item))
            layerNewName = invert_raster(item)
            raster_layers[index] = layerNewName
            
            for indexG, itemG in enumerate(raster_groups):
                for ix, itemX in enumerate(itemG):
                    if itemX == item:
                        raster_groups[indexG][ix] = layerNewName
                        break

    if (isInvertedLayer):

        logging.debug("There is an inverted raster. Recalculating linregress")

    #----------------------------------------------------
    # Remove previous extracted raster fields
    #----------------------------------------------------
    deleteUnnecessaryFields(
        feature_point,
        def_fields
    )

    #----------------------------------------------------
    # Build extraction list
    #----------------------------------------------------
    all_raster_extract_field_names = []

    for raster_path in raster_layers:

        raster_base = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        all_raster_extract_field_names.append(
            [raster_path, raster_base]
        )

    #----------------------------------------------------
    # Extract values from the updated raster list
    #----------------------------------------------------
    arcpy.sa.ExtractMultiValuesToPoints(
        in_point_features=feature_point,
        in_rasters=all_raster_extract_field_names,
        bilinear_interpolate_values="NONE"
    )

    #----------------------------------------------------
    # NOW read the fields that ArcGIS created
    #----------------------------------------------------
    existing_fields = [
        f.name for f in arcpy.ListFields(feature_point)
    ]

    raster_field_names = []

    for raster_path in raster_layers:

        raster_base = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        if raster_base in existing_fields:

            raster_field_names.append(raster_base)

        elif raster_base + "_1" in existing_fields:

            raster_field_names.append(raster_base + "_1")

        else:

            raise RuntimeError(
                "Extracted field not found: {}".format(raster_base)
            )

    # Re-read extracted values
    ray = arcpy.da.TableToNumPyArray(
        feature_point,
        raster_field_names
    )

    # Recalculate Pearson correlation after inversion
    resultCorrelationNp = np.zeros(
        len(raster_layers),
        dtype=[
            ('Features', 'U256'),
            ('corr', 'float'),
            ('sign', 'float')
        ]
    )

    for index, item in enumerate(raster_layers):

        x = np.array(
            ray[raster_field_names[index]]
        )

        y = np.array(array_y)

        res = linregress(y, x)

        resultCorrelationNp[index]['Features'] = (
            raster_field_names[index]
        )

        resultCorrelationNp[index]['corr'] = (
            res.rvalue
        )

        resultCorrelationNp[index]['sign'] = (
            res.pvalue
        )

    # -------------------------------------------------------------------
    # Prepare predictor matrix
    # -------------------------------------------------------------------

    X_all = pd.DataFrame(ray)

    # ----------------------------------------------------------
    # Clean ArcGIS field names
    # ----------------------------------------------------------

    X_all.columns = (
        X_all.columns
        .str.replace("_1$", "", regex=True)
        .str.replace("_Norm_uninv$", "", regex=True)
        .str.replace("_Norm_inv$", "", regex=True)
    )

    X_all.replace(
        [-9999, -99999],
        np.nan,
        inplace=True
    )

    X_all.dropna(
        inplace=True
    )

    # -------------------------------------------------------------------
    # Store Pearson results for ALL evaluated factors
    # -------------------------------------------------------------------

    factor_statistics = pd.DataFrame(
        resultCorrelationNp
    )

    factor_statistics["Factor"] = (
        factor_statistics["Features"]
        .str.replace("_1$", "", regex=True)
        .str.replace("_Norm_uninv$", "", regex=True)
        .str.replace("_Norm_inv$", "", regex=True)
    )

    factor_statistics.rename(
        columns={
            "corr": "Pearson_r",
            "sign": "P_value"
        },
        inplace=True
    )

    factor_statistics["Significance_Status"] = np.where(
        factor_statistics["P_value"] <= maxPValue,
        "Significant (P <= 0.001)",
        "Excluded (P > 0.001)"
    )

    logging.info("--------------------------------------")
    logging.info("X_all columns:")
    logging.info("{}".format(list(X_all.columns)))
    logging.info("--------------------------------------")

    ray = arcpy.da.TableToNumPyArray(
        feature_point,
        raster_field_names
    )

    resultCorrelationNp = np.zeros(len(raster_layers), dtype=[('Features', 'U256'), ('corr', 'float'), ('sign', 'float')])
    
    for index, item in enumerate(raster_layers):
        x = np.array(ray[raster_field_names[index]])
        y = np.array(array_y)
        res = linregress(y, x)
        resultCorrelationNp[index]['Features'] = raster_field_names[index]
        resultCorrelationNp[index]['corr'] = res.rvalue
        resultCorrelationNp[index]['sign'] = res.pvalue

    np.savetxt(
        os.path.join(results_directory, "factor_evaluation_result.csv"), resultCorrelationNp, fmt='%s, %.70f, %.70f' ,delimiter=",",header="Features,Pearson correlation (r),P value (significance)")
    logging.debug("End factor evaluation using Pearson Correlation")

# =================================================================================
# Step 4: Multicollinearity Analysis
# =================================================================================
    
    # -------------------------------------------------------------------
    # Pearson Correlation Matrix + Variance Inflation Factor (VIF)
    # -------------------------------------------------------------------

    logging.debug(" ")
    logging.debug("Step 4: Start Multicollinearity Analysis")

    # -------------------------------------------------------------------
    # Variance Inflation Factor threshold
    # -------------------------------------------------------------------
    VIF_THRESHOLD = 100

    # ----------------------------------------------------------
    # Keep only statistically significant predictors:
    # Only significant factors enter VIF
    # ----------------------------------------------------------

    significant_features = (
        factor_statistics
        .loc[
            factor_statistics["P_value"] <= maxPValue,
            "Factor"
        ]
        .tolist()
    )

    logging.info("--------------------------------------")
    logging.info(
        "Significant predictors retained (P <= {}): {}".format(
            maxPValue,
            len(significant_features)
        )
    )
    logging.info("--------------------------------------")

    # ----------------------------------------------------------
    # Check predictor names before entering VIF
    # ----------------------------------------------------------

    logging.info("X_all columns:")
    logging.info("{}".format(list(X_all.columns)))

    logging.info("--------------------------------------")
    logging.info("Significant features:")
    logging.info("{}".format(significant_features))

    missing_features = [
        f for f in significant_features
        if f not in X_all.columns
    ]

    logging.info("--------------------------------------")
    logging.info("Missing significant features:")
    logging.info("{}".format(missing_features))

    logging.info(
        "Number of X_all columns: {}".format(
            len(X_all.columns)
        )
    )

    logging.info(
        "Number of significant features: {}".format(
            len(significant_features)
        )
    )

    if missing_features:
        logging.error(
            "The following significant predictors are missing from X_all: {}".format(
                missing_features
            )
        )
        raise KeyError(
            "Significant predictors are not available in X_all."
        )

    # ----------------------------------------------------------
    # Create dataset containing only significant predictors
    # ----------------------------------------------------------

    X = X_all[significant_features]

    logging.info(
        "Samples used for multicollinearity analysis: {}".format(
            len(X)
        )
    )

    # -------------------------------------------------------------------
    # Variance Inflation Factor (VIF)
    # -------------------------------------------------------------------
    vif_table = pd.DataFrame()

    vif_table["Factor"] = X.columns

    vif_table["VIF"] = calculate_vif(X)


    vif_table["VIF_Status"] = np.where(
        vif_table["VIF"] <= VIF_THRESHOLD,
        "Acceptable (VIF <= {})".format(VIF_THRESHOLD),
        "Excluded (VIF > {})".format(VIF_THRESHOLD)
    )
    
    # -------------------------------------------------------------------
    # Save VIF Matrix for Retained and Excluded Factors
    # -------------------------------------------------------------------

    # Retained factors
    retained_vif_matrix = calculate_vif_matrix(
        X[vif_table.loc[
            vif_table["VIF"] <= VIF_THRESHOLD,
            "Factor"
        ]]
    )

    retained_vif_matrix_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Matrix_for_Retained_Factors.csv"
    )

    retained_vif_matrix.to_csv(
        retained_vif_matrix_csv,
        float_format="%.4f"
    )


    # Excluded factors
    excluded_factors = vif_table.loc[
        vif_table["VIF"] > VIF_THRESHOLD,
        "Factor"
    ].tolist()


    if len(excluded_factors) > 0:

        excluded_vif_matrix = calculate_vif_matrix(
            X[excluded_factors]
        )

        excluded_vif_matrix_csv = os.path.join(
            multicollinearity_directory,
            "VIF_Matrix_for_Excluded_Factors.csv"
        )

        excluded_vif_matrix.to_csv(
            excluded_vif_matrix_csv,
            float_format="%.4f"
        )


    logging.info(
        "VIF matrices successfully exported"
    )
    
    # -------------------------------------------------------------------
    # Identify correlated retained factors for excluded VIF factors
    # -------------------------------------------------------------------

    corr_matrix = X.corr()

    multicollinearity_links = []


    for factor in vif_table["Factor"]:

        status = vif_table.loc[
            vif_table["Factor"] == factor,
            "VIF_Status"
        ].values[0]


        if status == "Excluded (VIF > {})".format(VIF_THRESHOLD):

            related = []

            for other in corr_matrix.columns:

                if other != factor:

                    corr_value = abs(
                        corr_matrix.loc[
                            factor,
                            other
                        ]
                    )

                    # Only compare with retained significant factors
                    other_status = vif_table.loc[
                        vif_table["Factor"] == other,
                        "VIF_Status"
                    ]


                    if len(other_status) > 0:


                        if other_status.values[0] == "Acceptable (VIF <= {})".format(VIF_THRESHOLD):


                            other_vif = vif_table.loc[
                                vif_table["Factor"] == other,
                                "VIF"
                            ].values[0]


                            related.append(
                                "{} (VIF={:.2f}, r={:.3f})".format(
                                    other,
                                    other_vif,
                                    corr_value
                                )
                            )


            multicollinearity_links.append(
                "; ".join(related)
            )


        else:

            multicollinearity_links.append("")


    vif_table["Multicollinearity_with"] = (
        multicollinearity_links
    )
    
    # -------------------------------------------------------------------
    # Final Factor Assessment Report
    # -------------------------------------------------------------------

    final_factor_report = factor_statistics[
        [
            "Factor",
            "Pearson_r",
            "P_value",
            "Significance_Status"
        ]
    ].copy()


    final_factor_report = final_factor_report.merge(
        vif_table[
            [
                "Factor",
                "VIF",
                "VIF_Status",
                "Multicollinearity_with"
            ]
        ],
        on="Factor",
        how="left"
    )


    final_factor_report["Final_Decision"] = np.where(

        final_factor_report["P_value"] > maxPValue,

        "Excluded: Insignificant factor",

        np.where(
            final_factor_report["VIF"] > VIF_THRESHOLD,

            "Excluded: High multicollinearity",

            "Retained factor"
        )
    )



    final_report_csv = os.path.join(
        multicollinearity_directory,
        "Final_Factor_Assessment_Report.csv"
    )


    final_factor_report.to_csv(
        final_report_csv,
        index=False,
        float_format="%.6f"
    )


    logging.info(
        "Saved Final Factor Assessment Report"
    )

    logging.info(
        final_report_csv
    )
    
    # ------------------------------------------------
    # Build VIF Matrices (Diagonal Matrices)
    # ------------------------------------------------

    # These matrices are created after the VIF values are calculated.
    # Each factor's VIF is placed on the diagonal.

    retained_names = vif_table.loc[
        vif_table["VIF_Status"] == "Acceptable (VIF <= {})".format(VIF_THRESHOLD),
        "Factor"
    ].tolist()

    excluded_names = vif_table.loc[
        vif_table["VIF_Status"] == "Excluded (VIF > {})".format(VIF_THRESHOLD),
        "Factor"
    ].tolist()

    # ------------------------------------------------
    # Export VIF Table (NO ZERO VALUES)
    # ------------------------------------------------

    vif_csv = os.path.join(
        multicollinearity_directory,
        "Variance_Inflation_Factor.csv"
    )

    vif_table.to_csv(
        vif_csv,
        index=False,
        float_format="%.4f"
    )

    logging.info("Saved VIF table:")
    logging.info(vif_csv)


    # ------------------------------------------------
    # Pearson Correlation Matrices
    # ------------------------------------------------

    # ------------------------------------------------
    # Retained factors (VIF <= VIF_THRESHOLD)
    # ------------------------------------------------

    retained_names = vif_table.loc[
        vif_table["VIF"] <= VIF_THRESHOLD,
        "Factor"
    ].tolist()


    if len(retained_names) > 1:

        retained_corr = X[
            retained_names
        ].corr()

        retained_corr_csv = os.path.join(
            multicollinearity_directory,
            "Pearson_Correlation_Retained_Factors.csv"
        )

        retained_corr.to_csv(
            retained_corr_csv,
            float_format="%.4f"
        )

        logging.info(
            "Saved Pearson correlation matrix for retained factors"
        )


    # ------------------------------------------------
    # Excluded factors (VIF > VIF_THRESHOLD)
    # ------------------------------------------------

    excluded_names = vif_table.loc[
        vif_table["VIF"] > VIF_THRESHOLD,
        "Factor"
    ].tolist()


    if len(excluded_names) > 1:

        excluded_corr = X[
            excluded_names
        ].corr()

        excluded_corr_csv = os.path.join(
            multicollinearity_directory,
            "Pearson_Correlation_Excluded_Factors.csv"
        )

        excluded_corr.to_csv(
            excluded_corr_csv,
            float_format="%.4f"
        )

        logging.info(
            "Saved Pearson correlation matrix for excluded factors"
        )


    # ------------------------------------------------
    # Pearson Correlation Heatmaps
    # ------------------------------------------------


    def create_correlation_heatmap(
            corr_csv,
            output_jpg,
            title):


        corr_matrix = pd.read_csv(
            corr_csv,
            index_col=0
        )


        plt.figure(
            figsize=(12,10)
        )


        plt.imshow(
            corr_matrix.values,
            aspect="auto",
            vmin=-1,
            vmax=1
        )


        plt.colorbar(
            label="Pearson correlation (r)"
        )


        plt.xticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns,
            rotation=90
        )


        plt.yticks(
            range(len(corr_matrix.index)),
            corr_matrix.index
        )


        plt.title(title)

        plt.tight_layout()


        plt.savefig(
            output_jpg,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()



    # ------------------------------------------------
    # Retained factors heatmap
    # ------------------------------------------------

    retained_corr_csv = os.path.join(
        multicollinearity_directory,
        "Pearson_Correlation_Retained_Factors.csv"
    )


    if os.path.exists(retained_corr_csv):

        create_correlation_heatmap(
            retained_corr_csv,

            os.path.join(
                multicollinearity_directory,
                "Pearson_Heatmap_Retained_Factors.jpg"
            ),

            "Pearson Correlation Heatmap - Retained LSI Factors"
        )



    # ------------------------------------------------
    # Excluded factors heatmap
    # ------------------------------------------------

    excluded_corr_csv = os.path.join(
        multicollinearity_directory,
        "Pearson_Correlation_Excluded_Factors.csv"
    )


    if os.path.exists(excluded_corr_csv):

        create_correlation_heatmap(
            excluded_corr_csv,

            os.path.join(
                multicollinearity_directory,
                "Pearson_Heatmap_Excluded_Factors.jpg"
            ),

            "Pearson Correlation Heatmap - Excluded LSI Factors"
        )

    # ------------------------------------------------
    # Create Publication-Ready VIF Summary Table
    # ------------------------------------------------

    try:

        vif_summary_input = os.path.join(
            multicollinearity_directory,
            "Variance_Inflation_Factor.csv"
        )

        vif_df = pd.read_csv(
            vif_summary_input
        )

        # Sort factors according to VIF value
        vif_df = vif_df.sort_values(
            by="VIF",
            ascending=False
        )

        # Round VIF values
        vif_df["VIF"] = vif_df["VIF"].round(2)

        # Add multicollinearity interpretation
        def vif_interpretation(vif):

            if vif < 5:
                return "No significant multicollinearity"

            elif vif < 10:
                return "Moderate multicollinearity"

            else:
                return "High multicollinearity"

        vif_df["Multicollinearity_Assessment"] = (
            vif_df["VIF"]
            .apply(vif_interpretation)
        )

        # Add ranking
        vif_df.insert(
            0,
            "Rank",
            range(1, len(vif_df) + 1)
        )

        # Output file
        vif_summary_output = os.path.join(
            multicollinearity_directory,
            "VIF_Summary_Retained_Factors.csv"
        )

        vif_df.to_csv(
            vif_summary_output,
            index=False
        )

        print(
            "Publication-ready VIF summary saved:",
            vif_summary_output
        )

    except Exception as e:

        print(
            "Error creating VIF summary table:",
            e
        )
    
    # ------------------------------------------------
    # Export Final VIF Summary Report
    # ------------------------------------------------

    # Sort factors by VIF value (highest first)
    vif_summary = vif_table.sort_values(
        by="VIF",
        ascending=False
    )

    # Add VIF category interpretation
    def vif_category(value):
        if value < 5:
            return "No serious multicollinearity"
        elif value < 10:
            return "Moderate multicollinearity"
        else:
            return "High multicollinearity"

    vif_summary["VIF_Interpretation"] = vif_summary["VIF"].apply(vif_category)


    # Output CSV file
    vif_summary_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Summary_Report.csv"
    )

    vif_summary.to_csv(
        vif_summary_csv,
        index=False
    )

    print("VIF summary report saved:")
    print(vif_summary_csv)
  
    # ------------------------------------------------
    # Load Retained Factors for Final LSI Modelling
    # ------------------------------------------------

    # Read VIF summary table
    vif_summary = pd.read_csv(vif_summary_csv)


    # Select factors with acceptable VIF
    retained_factors = vif_summary.loc[
        vif_summary["VIF"] <= VIF_THRESHOLD,
        "Factor"
    ].tolist()


    # Save retained factor list
    retained_factor_csv = os.path.join(
        multicollinearity_directory,
        "Final_Retained_LSI_Factors.csv"
    )

    pd.DataFrame(
        {"Retained_Factors": retained_factors}
    ).to_csv(
        retained_factor_csv,
        index=False
    )


    print("-----------------------------------")
    print("Final retained LSI factors:")
    for factor in retained_factors:
        print(factor)

    print("-----------------------------------")
    print("Saved:")
    print(retained_factor_csv)
  
    # -------------------------------------------------------------------
    # Generate VIF Heatmaps for Retained and Excluded Factors
    # -------------------------------------------------------------------

    def create_vif_heatmap(
            vif_csv,
            output_jpg,
            title):


        vif_matrix = pd.read_csv(
            vif_csv,
            index_col=0
        )


        plt.figure(
            figsize=(10,8)
        )


        plt.imshow(
            vif_matrix.values,
            aspect="auto"
        )


        plt.colorbar(
            label="Variance Inflation Factor (VIF)"
        )


        plt.xticks(
            range(len(vif_matrix.columns)),
            vif_matrix.columns,
            rotation=90
        )


        plt.yticks(
            range(len(vif_matrix.index)),
            vif_matrix.index
        )


        plt.title(
            title
        )


        plt.tight_layout()


        plt.savefig(
            output_jpg,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


    # ------------------------------------------------
    # Retained factors VIF heatmap
    # ------------------------------------------------

    retained_vif_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Matrix_for_Retained_Factors.csv"
    )


    if os.path.exists(retained_vif_csv):

        create_vif_heatmap(
            retained_vif_csv,

            os.path.join(
                multicollinearity_directory,
                "VIF_Retained_Factors.jpg"
            ),

            "VIF Heatmap - Retained LSI Factors"
        )

    # ------------------------------------------------
    # Excluded factors VIF heatmap
    # ------------------------------------------------

    excluded_vif_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Matrix_for_Excluded_Factors.csv"
    )


    if os.path.exists(excluded_vif_csv):

        create_vif_heatmap(
            excluded_vif_csv,

            os.path.join(
                multicollinearity_directory,
                "VIF_Excluded_Factors.jpg"
            ),

            "VIF Heatmap - Excluded High Multicollinearity Factors"
        )


    logging.info(
        "VIF heatmaps generated successfully"
    )
  
    # -------------------------------------------------------------------
    # Generate VIF Summary Statistics and Interpretation
    # -------------------------------------------------------------------

    # Input VIF CSV
    vif_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Matrix_for_Retained_Factors.csv"
    )

    # Output summary CSV
    vif_summary_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Summary_Retained_Factors.csv"
    )

    # Read VIF matrix
    vif_matrix = pd.read_csv(
        vif_csv,
        index_col=0
    )

    # Extract diagonal VIF values
    vif_values = pd.DataFrame({
        "Factor": vif_matrix.index,
        "VIF": vif_matrix.values.diagonal()
    })
     
    # Save summary before using it
    vif_values.to_csv(
        vif_summary_csv,
        index=False,
        float_format="%.4f"
    )
    
    # -------------------------------------------------------------------
    # Automatic VIF Threshold Checking Before LSI Modelling
    # -------------------------------------------------------------------

    # User-defined VIF threshold
    # Common thresholds:
    # VIF <= threshold : Acceptable (retain factor)
    # VIF > threshold  : Potential severe multicollinearity (excluded factor)


    # Input VIF summary table
    vif_summary_file = os.path.join(
        multicollinearity_directory,
        "VIF_Summary_Retained_Factors.csv"
    )


    # Output flagged factors
    vif_check_file = os.path.join(
        multicollinearity_directory,
        "VIF_Threshold_Check_Report.csv"
    )


    # Use the VIF summary already created in memory
    vif_summary = vif_values.copy()
 

    # ---------------------------------------------------
    # VIF threshold evaluation
    # ---------------------------------------------------

    vif_summary["VIF_Status"] = np.where(
        vif_summary["VIF"] <= VIF_THRESHOLD,
        "Acceptable (VIF <= {})".format(VIF_THRESHOLD),
        "Review Required (VIF > VIF_THRESHOLD)"
    )

    # ---------------------------------------------------
    # Identify factors exceeding threshold
    # ---------------------------------------------------

    high_vif_factors = vif_summary[
        vif_summary["VIF"] > VIF_THRESHOLD
    ]


    # Save VIF checking report
    vif_summary.to_csv(
        vif_check_file,
        index=False
    )

    # ---------------------------------------------------
    # Console summary
    # ---------------------------------------------------

    print("\n---------------------------------------")
    print("VIF THRESHOLD CHECK FOR LSI FACTORS")
    print("---------------------------------------")

    print(
        "Applied VIF Threshold:",
        VIF_THRESHOLD
    )

    print(
        "Total retained factors checked:",
        len(vif_summary)
    )

    print(
        "Factors with VIF <= {}:".format(VIF_THRESHOLD),
        len(vif_summary[vif_summary["VIF"] <= VIF_THRESHOLD])
    )

    print(
        "Factors with VIF > {}:".format(VIF_THRESHOLD),
        len(high_vif_factors)
    )


    if len(high_vif_factors) > 0:

        print("\nFactors requiring review:")

        print(
            high_vif_factors[
                ["Factor", "VIF"]
            ]
        )

    else:

        print(
            "\nAll retained factors have acceptable VIF values (VIF <= {}).".format(
                VIF_THRESHOLD
            )
        )

    print("---------------------------------------")


    # ---------------------------------------------------
    # Classify multicollinearity level
    # ---------------------------------------------------

    def vif_classification(vif):

        if vif < 3:
            return "No significant multicollinearity"

        elif vif < 5:
            return "Moderate multicollinearity"

        if vif <= 10:
            return "High multicollinearity"

        else:
            return "Very high multicollinearity"


    vif_values["Interpretation"] = (
        vif_values["VIF"]
        .apply(vif_classification)
    )

    # ---------------------------------------------------
    # Overall statistics
    # ---------------------------------------------------

    summary_statistics = pd.DataFrame({

        "Statistic": [
            "Number of retained factors",
            "Minimum VIF",
            "Maximum VIF",
            "Mean VIF"
        ],

        "Value": [
            len(vif_values),
            round(vif_values["VIF"].min(), 2),
            round(vif_values["VIF"].max(), 2),
            round(vif_values["VIF"].mean(), 2)
        ]
    })


    # Save detailed VIF table
    vif_values.to_csv(
        vif_summary_csv,
        index=False
    )

    # Save overall statistics
    summary_statistics_csv = os.path.join(
        multicollinearity_directory,
        "VIF_Overall_Statistics.csv"
    )

    summary_statistics.to_csv(
        summary_statistics_csv,
        index=False
    )


    print("VIF summary saved:")
    print(vif_summary_csv)

    print("Overall VIF statistics saved:")
    print(summary_statistics_csv)


    vif_csv = os.path.join(
        multicollinearity_directory,
        "Variance_Inflation_Factor.csv"
    )


    vif_table.to_csv(
        vif_csv,
        index=False,
        float_format="%.4f"
    )

    logging.info("Saved VIF Table")
    logging.info(vif_csv)


    # -------------------------------------------------------------------
    # Maintained and Excluded Factors
    # -------------------------------------------------------------------
    kept_features = vif_table.loc[
        vif_table["VIF_Status"] == "Acceptable (VIF <= {})".format(VIF_THRESHOLD),
        "Factor"
    ].tolist()


    excluded_features = vif_table.loc[
        vif_table["VIF_Status"] == "Excluded (VIF > {})".format(VIF_THRESHOLD),
        "Factor"
    ].tolist()


    # -------------------------------------------------------------------
    # VIF Plot
    # -------------------------------------------------------------------
    plt.figure(figsize=(12,6))


    plt.bar(
        vif_table["Factor"],
        vif_table["VIF"]
    )


    plt.axhline(
        y=VIF_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label="VIF = {}".format(VIF_THRESHOLD)
    )


    plt.xticks(
        rotation=90,
        fontsize=8
    )

    plt.ylabel("Variance Inflation Factor")

    plt.legend()

    plt.tight_layout()


    vif_plot = os.path.join(
        multicollinearity_directory,
        "VIF.jpg"
    )


    plt.savefig(
        vif_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logging.info("Saved VIF Plot")
    logging.info(vif_plot)


    # ================================================================
    # Step 5: Final Factor Selection for LSI Modelling
    #
    # A factor is retained ONLY if:
    #   1. P-value <= maxPValue
    #   2. VIF <= VIF_THRESHOLD
    #
    # Factors excluded by either criterion are removed before
    # linear regression and LSI construction.
    # ================================================================

    logging.info("--------------------------------")
    logging.info("Step 5: FINAL FACTOR SELECTION FOR LSI")
    logging.info("--------------------------------")

    # ---------------------------------------------------------------
    # Helper function:
    # Convert normalized raster filename to the factor name used
    # in factor_statistics and vif_table.
    #
    # Examples:
    # Annual_Precipitation_Norm_uninv
    # Annual_Precipitation_Norm_inv
    #
    # become:
    # Annual_Precipitation_Norm_uninv
    # Annual_Precipitation_Norm_inv
    #
    # We keep the complete normalized name because this is the
    # name stored in the statistical tables.
    # ---------------------------------------------------------------

    def get_raster_factor_name(raster_path):

        name = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        # Remove normalization/inversion suffixes so that
        # the original factor name is used for statistical
        # selection and raster processing.
        name = name.replace("_Norm_uninv", "")
        name = name.replace("_Norm_inv", "")

        return name


    # ---------------------------------------------------------------
    # Build lookup tables
    # ---------------------------------------------------------------

    significance_lookup = {}

    for _, row in factor_statistics.iterrows():

        factor_name = str(row["Factor"])

        significance_lookup[factor_name] = {
            "P_value": float(row["P_value"]),
            "Significant": float(row["P_value"]) <= maxPValue
        }


    vif_lookup = {}

    for _, row in vif_table.iterrows():

        factor_name = str(row["Factor"])

        vif_lookup[factor_name] = {
            "VIF": float(row["VIF"]),
            "VIF_Acceptable": float(row["VIF"]) <= VIF_THRESHOLD
        }


    # ---------------------------------------------------------------
    # Determine final statistically retained factors
    # ONLY significant factors with acceptable VIF are retained.
    # ---------------------------------------------------------------

    final_retained_factors = []

    significance_excluded_factors = []
    vif_excluded_factors = []

    for factor_name in significance_lookup:

        p_value = significance_lookup[factor_name]["P_value"]

        # -----------------------------------------------------------
        # First criterion: significance
        # -----------------------------------------------------------

        if p_value > maxPValue:

            significance_excluded_factors.append(
                factor_name
            )

            logging.info(
                "{} EXCLUDED: P-value = {:.6g} > {:.6g}".format(
                    factor_name,
                    p_value,
                    maxPValue
                )
            )

            continue

        # -----------------------------------------------------------
        # Second criterion: VIF
        # -----------------------------------------------------------

        if factor_name not in vif_lookup:

            logging.warning(
                "{} is significant but was not found in VIF table. "
                "Excluded from final LSI.".format(
                    factor_name
                )
            )

            vif_excluded_factors.append(
                factor_name
            )

            continue

        vif_value = vif_lookup[factor_name]["VIF"]

        if vif_value > VIF_THRESHOLD:

            vif_excluded_factors.append(
                factor_name
            )

            logging.info(
                "{} EXCLUDED: VIF = {:.3f} > {:.3f}".format(
                    factor_name,
                    vif_value,
                    VIF_THRESHOLD
                )
            )

            continue

        # -----------------------------------------------------------
        # Passed BOTH criteria
        # -----------------------------------------------------------

        final_retained_factors.append(
            factor_name
        )

        logging.info(
            "{} RETAINED: P-value = {:.6g}, VIF = {:.3f}".format(
                factor_name,
                p_value,
                vif_value
            )
        )


    # -----------------------------------------------------------
    # Summary: Factor Significance and VIF Test
    # -----------------------------------------------------------

    logging.info("--------------------------------")
    logging.info(
        "Final retained factors: {}".format(
            len(final_retained_factors)
        )
    )

    logging.info(
        "Excluded by significance test: {}".format(
            len(significance_excluded_factors)
        )
    )

    logging.info(
        "Excluded by VIF test: {}".format(
            len(vif_excluded_factors)
        )
    )

    logging.info("--------------------------------")


    # ---------------------------------------------------------------
    # Remove all factors that did NOT pass BOTH tests
    # ---------------------------------------------------------------

    logging.info(
        "Removing factors excluded by significance and/or VIF..."
    )

    # ---------------------------------------------------------------
    # Loop through all raster groups
    # ---------------------------------------------------------------

    for group_index, group in enumerate(raster_groups):

        # Use copy because the original group is modified
        for raster_name in group[:]:

            raster_factor_name = get_raster_factor_name(
                raster_name
            )

            # -------------------------------------------------------
            # Factor is NOT in the final retained list
            # -------------------------------------------------------

            if raster_factor_name not in final_retained_factors:

                # Find statistical information where available
                p_value = None
                vif_value = None

                if raster_factor_name in significance_lookup:

                    p_value = significance_lookup[
                        raster_factor_name
                    ]["P_value"]

                if raster_factor_name in vif_lookup:

                    vif_value = vif_lookup[
                        raster_factor_name
                    ]["VIF"]

                # ---------------------------------------------------
                # Determine reason for exclusion
                # ---------------------------------------------------

                if (
                    p_value is not None
                    and
                    p_value > maxPValue
                ):

                    logging.info(
                        "Removed {} from {}: "
                        "excluded by significance test "
                        "(P = {:.6g} > {:.6g})".format(
                            raster_factor_name,
                            group_names[group_index],
                            p_value,
                            maxPValue
                        )
                    )

                elif (
                    vif_value is not None
                    and
                    vif_value > VIF_THRESHOLD
                ):

                    logging.info(
                        "Removed {} from {}: "
                        "excluded by VIF "
                        "(VIF = {:.3f} > {:.3f})".format(
                            raster_factor_name,
                            group_names[group_index],
                            vif_value,
                            VIF_THRESHOLD
                        )
                    )

                else:

                    logging.info(
                        "Removed {} from {}: "
                        "did not pass final factor-selection criteria.".format(
                            raster_factor_name,
                            group_names[group_index]
                        )
                    )

                # ---------------------------------------------------
                # IMPORTANT: Actually remove the raster from the group
                # ---------------------------------------------------

                group.remove(
                    raster_name
                )

    # ---------------------------------------------------------------
    # Update raster list after FINAL factor filtering
    # ---------------------------------------------------------------

    raster_layers = list(
        chain.from_iterable(
            raster_groups
        )
    )

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    if len(raster_layers) == 0:

        logging.error(
            "No factors remain after significance and VIF filtering."
        )

        raise RuntimeError(
            "LSI cannot be generated because no predictors passed "
            "both the significance and VIF criteria."
        )

    # ---------------------------------------------------------------
    # Print final factors
    # ---------------------------------------------------------------

    print("\n-----------------------------------")
    print("FINAL LSI FACTORS")
    print("-----------------------------------")

    for factor in raster_layers:

        print(
            os.path.splitext(
                os.path.basename(factor)
            )[0]
        )

    print("-----------------------------------")
    print(
        "Number of final LSI factors:",
        len(raster_layers)
    )
    print("-----------------------------------")


    logging.info("--------------------------------")
    logging.info(
        "FINAL LSI FACTOR COUNT: {}".format(
            len(raster_layers)
        )
    )

    logging.info(
        "Final LSI factors: {}".format(
            [
                os.path.splitext(
                    os.path.basename(f)
                )[0]
                for f in raster_layers
            ]
        )
    )

    logging.info("--------------------------------")


    # ---------------------------------------------------------------
    # Rebuild extraction field names for FINAL retained factors
    # ---------------------------------------------------------------

    raster_field_names = []

    for raster_path in raster_layers:

        raster_base = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        raster_field_names.append(
            raster_base
        )


    print("\nFinal fields requested:")

    print(
        raster_field_names
    )

    # ---------------------------------------------------------------
    # Extract only FINAL retained factors
    # ---------------------------------------------------------------

    ray = arcpy.da.TableToNumPyArray(
        feature_point,
        raster_field_names
    )

    # Update raster list after final significance + VIF filtering
    raster_layers = list(chain.from_iterable(raster_groups))


    raster_field_names = []

    for raster_path in raster_layers:

        raster_base = os.path.splitext(
            os.path.basename(raster_path)
        )[0]

        raster_field_names.append(
            raster_base
        )

    print("\nFields in feature class:")
    for f in arcpy.ListFields(feature_point):
        print(f.name)

    print("\nRequested fields:")
    print(raster_field_names)
    
    ray = arcpy.da.TableToNumPyArray(
        feature_point,
        raster_field_names
    )
    
    # ===============================================================
    # Step 6: Linear regression (Eg: c3, c2 calculations)
    # ===============================================================
    logging.debug("Step 6: Start Linear regression")
    result = []
    for index, item in enumerate(raster_field_names):
        #print("index:",index, "item:",item)
        x = np.array(ray[item])
        y = np.array(array_y)
        z = np.polyfit(x,y, 3)
        result.append([str(item),z[0],z[1],z[2],z[3]])

    npResult = np.array(result)
 
    resultLinearNp = np.zeros(len(result), dtype=[('Features', 'U256'), ('c3', 'float'), ('c2', 'float'), ('c1', 'float'), ('b', 'float')])
    resultLinearNp['Features'] = raster_layers#npResult[:, 0]
    resultLinearNp['c3'] = npResult[:, 1]
    resultLinearNp['c2'] = npResult[:, 2]
    resultLinearNp['c1'] = npResult[:, 3]
    resultLinearNp['b'] = npResult[:, 4]
 
    np.savetxt(
        os.path.join(results_directory, "result_linear.csv"), resultLinearNp, fmt='%s, %.11f, %.11f, %.11f, %.11f' ,delimiter=",",header="Features,c3,c2,c1,b")

    for index, item in enumerate(raster_layers):
        val = result[index]
        c3 = val[1]
        c2 = val[2]
        c1 = val[3]
        b = val[4]
        logging.debug(" ".join(["processing item",item,"c3",str(c3),"c2",str(c2),"c1",str(c1),"b",str(b)]))
        myRaster = arcpy.sa.Raster(item)

        # LSIf Calculation
        rasterOut = 1 / (1 + arcpy.sa.Exp((c3 * (myRaster ** 3) + (c2 * myRaster ** 2) + (c1 * myRaster + b)) * -1))

        factor_output = os.path.join(
            working_directory_output,
            os.path.splitext(os.path.basename(item))[0] + "_Factor.tif"
        )

        rasterOut.save(factor_output)
        
        logging.debug("Saved raster "+item)

    logging.debug("End Linear regression")

    # ===========================================================================
    # Step 7: Generating LSI by combining all LSIf into one raster layer
    # ===========================================================================
    logging.debug("Step 7: Start Generating LSI")

    rasterTotal = None

    cnt = 0
    myloopR = 0
    
    for rg in raster_groups:
        logging.debug("===New Group====")
        rasterGroup = None
        myloop = 0
       
        numOfGroupElement = 0
        
        for r in rg:

            factor_name = os.path.splitext(
                os.path.basename(r)
            )[0]

            # -------------------------------------------------------
            # IMPORTANT:
            # Only FINAL retained factors should reach this stage.
            # They have already passed:
            #
            # P-value <= maxPValue
            # AND
            # VIF <= VIF_THRESHOLD
            # -------------------------------------------------------

            raster_file_name = os.path.splitext(
                os.path.basename(r)
            )[0]

            factor_name = get_raster_factor_name(r)

            # -------------------------------------------------------
            # Only final retained factors are allowed into LSI
            # -------------------------------------------------------

            if factor_name not in final_retained_factors:

                logging.debug(
                    "Layer {} excluded from LSI because it "
                    "did not pass final significance/VIF filtering.".format(
                        factor_name
                    )
                )

                continue

            factor_raster = os.path.join(
                working_directory_output,
                raster_file_name + "_Factor.tif"
            )

            if not arcpy.Exists(factor_raster):

                logging.warning(
                    "Factor raster does not exist: {}".format(
                        factor_raster
                    )
                )

                continue
                
            # -------------------------------------------------------
            # Obtain Pearson R-value
            # -------------------------------------------------------

            valueR = None

            for item2 in resultCorrelationNp:

                correlation_factor = str(item2[0])

                correlation_factor = correlation_factor.replace(
                    "_Norm_uninv",
                    ""
                )

                correlation_factor = correlation_factor.replace(
                    "_Norm_inv",
                    ""
                )

                correlation_factor = correlation_factor.replace(
                    "_1",
                    ""
                )

                if correlation_factor == factor_name:

                    valueR = float(item2[1])

                    break

            # -------------------------------------------------------
            # Safety check
            # -------------------------------------------------------

            if valueR is None:

                logging.warning(
                    "No Pearson R-value found for final retained "
                    "factor: {}".format(
                        factor_name
                    )
                )

                continue

            # -------------------------------------------------------
            # Add factor to group
            # -------------------------------------------------------

            inraster = arcpy.sa.Raster(
                factor_raster
            )

            if rasterGroup is None:

                rasterGroup = (
                    inraster * valueR
                )

            else:

                rasterGroup += (
                    inraster * valueR
                )

            numOfGroupElement += 1

            myloop += 1


            logging.debug(
                "{}_Factor.tif * {} <- R-value "
                "(FINAL RETAINED FACTOR)".format(
                    factor_name,
                    valueR
                )
            )
            
        
        #At least one layer element should be processed for creating result layers
        if(numOfGroupElement > 0):

            rasterGroup = arcpy.sa.Divide(
                rasterGroup,
                numOfGroupElement
            )

            output_group = os.path.join(
                results_directory,
                group_names[cnt] + ".tif"
            )

            rasterGroup.save(output_group)

            logging.debug(
                "Group Output: {}  LEN:{}".format(
                    output_group,
                    numOfGroupElement
                )
            )

            if rasterTotal is None:

                rasterTotal = rasterGroup

            else:

                rasterTotal = rasterTotal + rasterGroup

            myloopR += 1

        else:
            logging.debug("No layer in this Group")   
        cnt+=1
    logging.debug("End Finding LSI")

    
    # -------------------------------------------------------
    # Save LSI Index Raster
    # -------------------------------------------------------

    logging.debug(
        "Preparing final LSI_INDEX raster..."
    )


    # Check if any raster group was successfully produced

    if rasterTotal is None or myloopR == 0:

        logging.error(
            "No valid raster groups available for LSI calculation."
        )

        raise RuntimeError(
            "LSI_INDEX.tif cannot be generated because no factors remain."
        )


    # Average all group outputs
    rasterTotal = arcpy.sa.Divide(
        rasterTotal,
        myloopR
    )


    # Output LSI index
    lsi_index_output = os.path.join(
        results_directory,
        "LSI_INDEX.tif"
    )


    # Remove old file if exists
    if arcpy.Exists(lsi_index_output):

        arcpy.management.Delete(
            lsi_index_output
        )


    # Save LSI_INDEX.tif
    rasterTotal.save(
        lsi_index_output
    )


    logging.info(
        "LSI_INDEX.tif successfully generated:"
    )

    logging.info(
        lsi_index_output
    )


    logging.info(
        "Number of contributing raster groups: {}".format(
            myloopR
        )
    )
    
    # ===========================================================================
    # Step 8: LSI Normalization
    # ===========================================================================
    logging.debug(" ")
    logging.debug("Step 8: Start LSI Normalization")
    
    lsi_max = float(arcpy.management.GetRasterProperties(rasterTotal,"MAXIMUM").getOutput(0))
    lsi_min = float(arcpy.management.GetRasterProperties(rasterTotal,"MINIMUM").getOutput(0))

    logging.debug("MINIMUM value " +str(lsi_min))
    logging.debug("MAXIMUM value " +str(lsi_max))
    logging.debug("MEAN value " +arcpy.management.GetRasterProperties(rasterTotal,"MEAN").getOutput(0))
    logging.debug("STD value " + arcpy.management.GetRasterProperties(rasterTotal,"STD").getOutput(0))
    
    #normalizedLSI = arcpy.Raster(rasterTotal.getRasterInfo())
    xa = rasterTotal - lsi_min

    LSI_Normalized = arcpy.sa.Divide(
        xa,
       (lsi_max - lsi_min)
    )

    # ----------------------------------------------------
    # Save normalized LSI
    # ----------------------------------------------------
    lsi_output = os.path.join(
        results_directory,
        "LSI_Normalized.tif"
    )

    if arcpy.Exists(lsi_output):
        arcpy.management.Delete(lsi_output)

    if arcpy.Exists(lsi_output):

        arcpy.management.Delete(
            lsi_output
        )


    LSI_Normalized.save(
        lsi_output
    )


    logging.info(
        "LSI_Normalized.tif created successfully"
    )

    logging.info("--------------------------------")
    logging.info("Normalized LSI saved")
    logging.info(lsi_output)
    logging.info("--------------------------------")
    
except Exception as ex:

    logging.error(traceback.format_exc())
    logging.error(ex)

    raise SystemExit(
        "Program terminated due to model failure."
    )

# ===================================================================================
# Step 9: Validation of Normalized LSI using ROC/AUC + Pearson Correlation
# ===================================================================================

lsi_raster = lsi_output

if not arcpy.Exists(lsi_raster):

    raise SystemExit(
        "LSI_Normalized.tif does not exist"
    )
    
# -------------------------------------------------------------------
# Validation using ROC/AUC + Pearson Correlation
# -------------------------------------------------------------------

logging.debug(" ")
logging.debug("Step 9: Start Validation (ROC + Pearson)")

# -------------------------------------------------------------------
# Remove previous extracted fields
# -------------------------------------------------------------------
deleteUnnecessaryFields(feature_point_LSI_norm, def_fields)

# -------------------------------------------------------------------
# Extract LSI values to validation points
# -------------------------------------------------------------------
arcpy.sa.ExtractMultiValuesToPoints(
    in_point_features=feature_point_LSI_norm,
    in_rasters=[[lsi_raster, "LSI_Normalized"]],
    bilinear_interpolate_values="NONE"
)

# -------------------------------------------------------------------
# Read validation data
# -------------------------------------------------------------------
y_true = []
y_score = []

with arcpy.da.SearchCursor(
        feature_point_LSI_norm,
        ["LS1NOLS0", "LSI_Normalized"]) as cursor:

    for row in cursor:
        if row[0] is not None and row[1] is not None:
            y_true.append(int(row[0]))
            y_score.append(float(row[1]))

y_true = np.array(y_true)
y_score = np.array(y_score)

logging.info("Validation samples: {}".format(len(y_true)))

# -------------------------------------------------------------------
# ROC Calculation (manual, no sklearn)
# -------------------------------------------------------------------

thresholds = np.sort(np.unique(y_score))[::-1]

tpr_list = []
fpr_list = []

P = np.sum(y_true == 1)
N = np.sum(y_true == 0)

for t in thresholds:

    y_pred = (y_score >= t).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    TPR = TP / P if P > 0 else 0
    FPR = FP / N if N > 0 else 0

    tpr_list.append(TPR)
    fpr_list.append(FPR)

tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)


# -------------------------------------------------------------------
# Save all ROC coordinates
# -------------------------------------------------------------------

roc_table = pd.DataFrame({
    "Threshold": thresholds,
    "True_Positive_Rate_TPR": tpr_list,
    "False_Positive_Rate_FPR": fpr_list
})

roc_table = roc_table.sort_values(
    by="False_Positive_Rate_FPR"
).reset_index(drop=True)

# Add ROC origin
roc_table = pd.concat([
    pd.DataFrame({
        "Threshold": [np.inf],
        "True_Positive_Rate_TPR": [0.0],
        "False_Positive_Rate_FPR": [0.0]
    }),
    roc_table
], ignore_index=True)

# Add ROC endpoint
roc_table = pd.concat([
    roc_table,
    pd.DataFrame({
        "Threshold": [-np.inf],
        "True_Positive_Rate_TPR": [1.0],
        "False_Positive_Rate_FPR": [1.0]
    })
], ignore_index=True)

roc_csv_file = os.path.join(
    results_directory,
    "ROC_TPR_FPR_All_Points.csv"
)

roc_table.to_csv(
    roc_csv_file,
    index=False,
    float_format="%.8f"
)

logging.info("--------------------------------")
logging.info("ROC coordinate table saved:")
logging.info(roc_csv_file)
logging.info("--------------------------------")

# -------------------------------------------------------------------
# AUC (trapezoidal rule)
# -------------------------------------------------------------------

order = np.argsort(fpr_list)

fpr_sorted = fpr_list[order]
tpr_sorted = tpr_list[order]

fpr_sorted = np.insert(fpr_sorted, 0, 0)
tpr_sorted = np.insert(tpr_sorted, 0, 0)

fpr_sorted = np.append(fpr_sorted, 1)
tpr_sorted = np.append(tpr_sorted, 1)

roc_auc = np.trapz(
    tpr_sorted,
    fpr_sorted
)

# -------------------------------------------------------------------
# Pearson correlation
# -------------------------------------------------------------------
pearson_r = np.corrcoef(y_score, y_true)[0, 1]

logging.info("--------------------------------")
logging.info("Pearson r = {:.4f}".format(pearson_r))
logging.info("--------------------------------")

# -------------------------------------------------------------------
# Save CSV (summary)
# -------------------------------------------------------------------
csv_file = os.path.join(
    results_directory,
    "LSI_Validation_Results.csv"
)

with open(csv_file, "w") as f:
    f.write("Metric,Value\n")
    f.write("ROC_AUC,{}\n".format(roc_auc))
    f.write("Pearson_r,{}\n".format(pearson_r))
    f.write("Validation_samples,{}\n".format(len(y_true)))

logging.info("Saved validation CSV:")
logging.info(csv_file)

# -------------------------------------------------------------------
# Plot ROC Curve
# -------------------------------------------------------------------
plt.figure(figsize=(7, 7))

plt.plot(
    fpr_list,
    tpr_list,
    linewidth=2,
    label="ROC (AUC = {:.4f})".format(roc_auc)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random"
)

plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSI Validation")
plt.grid(True)
plt.legend(loc="lower right")

jpg_file = os.path.join(
    results_directory,
    "ROC_AUC_Validation.jpg"
)

plt.savefig(jpg_file, dpi=300, bbox_inches="tight")

plt.close()

logging.info("Saved ROC JPG:")
logging.info(jpg_file)

logging.debug("End Validation")

#Release Extension
arcpy.CheckInExtension("Spatial")