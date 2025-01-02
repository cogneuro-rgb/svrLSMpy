from nilearn.plotting import view_img
import tempita
from pathlib import Path
import pandas as pd
import nibabel as nib
import base64

from modules.plot_axial_slices import get_axial_slices,save_axial_mosaic

def save_report(output_file, svr_params, behaviour_name, n_permutations, alpha, zmap_range, zmap, min_patient_count,num_patients,covariate_info, nifti_zmap, zmap_atlas_output_dir, time_taken, num_lesions, mean_lesion_volume, n_clusters=5):
    """
    Save a comprehensive report of the LSM analysis, including parameters, significant voxels, and visualization.
    """
    print("Saving report...")

    atlas_reader_output_folder = Path(zmap_atlas_output_dir)
    cluster_csv_path = atlas_reader_output_folder / "atlasreader_clusters.csv"
    fileExists = False
    try:
        cluster_df = pd.read_csv(cluster_csv_path)
        print("File atlasreader_clusters.csv loaded successfully!")
        if (cluster_df.shape[0] < 5):
            n_clusters = int(cluster_df.shape[0])

        top_clusters = cluster_df.nlargest(n_clusters, 'volume_mm')  # Get top 5 clusters by volume

        overview_image_path = atlas_reader_output_folder / "atlasreader.png"
        cluster_image_paths = [atlas_reader_output_folder / f"atlasreader_cluster0{i}.png" for i in
                               range(1, n_clusters + 1)]

        def encode_image(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')

        overview_image_base64 = encode_image(overview_image_path)
        cluster_images_base64 = [encode_image(path) for path in cluster_image_paths]

        fileExists = True
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print("Error: The file 'atlasreader_clusters.csv' was not found.")
    except Exception as e:
        # Catch any other exceptions
        print(f"An unexpected error occurred: {e}")

    #significant_voxels = np.abs(zmap) > np.percentile(np.abs(zmap), 100 * (1 - alpha))

    output_folder = Path(output_file).parent
    html_view = view_img(f"{output_folder}/zmap.nii.gz",threshold=1.7,black_bg=False,cmap="jet")


    lesion_overlap_path = f"{output_folder}/lesion_overlap.nii.gz"
    lesion_overlap_mosaic_path = f"{output_folder}/lesion_overlap_mosaic.png"

    num_slices = 10
    cut_coords = get_axial_slices(lesion_overlap_path, num_slices)
    save_axial_mosaic(lesion_overlap_path, cut_coords, lesion_overlap_mosaic_path)


    lesion_overlap_filtered_path = f"{output_folder}/lesion_overlap_filtered.nii.gz"
    lesion_overlap_filtered_mosaic_path = f"{output_folder}/lesion_overlap_filtered_mosaic.png"

    save_axial_mosaic(lesion_overlap_filtered_path, cut_coords, lesion_overlap_filtered_mosaic_path)


    svr_beta_map_path = f"{output_folder}/beta_map.nii.gz"
    svr_beta_map_mosaic_path = f"{output_folder}/svr_beta_map_mosaic.png"

    cut_coords = get_axial_slices(svr_beta_map_path, num_slices)
    save_axial_mosaic(svr_beta_map_path, cut_coords, svr_beta_map_mosaic_path)


    zmap_path = f"{output_folder}/zmap.nii.gz"
    zmap_mosaic_path = f"{output_folder}/zmap_mosaic.png"

    save_axial_mosaic(zmap_path, cut_coords, zmap_mosaic_path)

    zmap_p05_path = f"{output_folder}/zmap_p05.nii.gz"
    zmap_p05_mosaic_path = f"{output_folder}/zmap_p05_mosaic.png"

    save_axial_mosaic(zmap_p05_path, cut_coords, zmap_p05_mosaic_path)

    zmap_p01_path = f"{output_folder}/zmap_p01.nii.gz"
    zmap_p01_mosaic_path = f"{output_folder}/zmap_p01_mosaic.png"

    save_axial_mosaic(zmap_p01_path, cut_coords, zmap_p01_mosaic_path)

    zmap_p005_path = f"{output_folder}/zmap_p005.nii.gz"
    zmap_p005_mosaic_path = f"{output_folder}/zmap_p005_mosaic.png"

    save_axial_mosaic(zmap_p005_path, cut_coords, zmap_p005_mosaic_path)

    zmap_p001_path = f"{output_folder}/zmap_p001.nii.gz"
    zmap_p001_mosaic_path = f"{output_folder}/zmap_p001_mosaic.png"

    save_axial_mosaic(zmap_p001_path, cut_coords, zmap_p001_mosaic_path)


    with open(output_file, 'w') as f:
        f.write(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVR-Based Lesion-Symptom Mapping Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7fc;
                color: #333;
                margin: 20px;
                line-height: 1.7;
            }}
            h1 {{
                color: #004d99;
                font-size: 2.5em;
                margin-bottom: 20px;
            }}
            h2 {{
                color: #004d99;
                font-size: 1.8em;
                border-bottom: 3px solid #004d99;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h3 {{
                font-size: 1.5em;
                margin-top: 20px;
                color: #2c3e50;
            }}
            p {{
                font-size: 1.1em;
                margin: 15px 0;
                color: #34495e;
            }}
            ul {{
                padding-left: 20px;
                font-size: 1.1em;
            }}
            li {{
                margin-bottom: 10px;
            }}
            .container {{
                width: 85%;
                margin: auto;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #eaf2f8;
                padding: 15px;
                border-left: 5px solid #1e88e5;
            }}
            .result-table {{
                width: 100%;
                margin-top: 20px;
                border-collapse: collapse;
            }}
            .result-table th, .result-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .result-table th {{
                background-color: #f4f7fc;
                font-weight: bold;
            }}
            .result-table td {{
                background-color: #fafafa;
            }}
            a {{
                color: #004d99;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .footer {{
                text-align: center;
                font-size: 0.9em;
                color: #777;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>svrLSMpy Report for {behaviour_name}</h1>

            <div class="section">
                <h2>Methodology</h2>
                <p>This report is an analysis of the relationship between lesion status and the behavioral score {behaviour_name}; performing lesion-symptom mapping using support vector regression. {num_patients} binary lesion files in MNI space were analyzed, including only voxels with at least {min_patient_count} overlapping lesions. Lesion volume was controlled using vector normalization. Covariates were regressed out of the behavioural scores. Support Vector Regression (SVR) was applied with parameters gamma = {svr_params['gamma']}, cost = {svr_params['C']}, and epsilon = {svr_params['epsilon']}, employing grid search optimization. Z maps were derived from on null distributions based on {n_permutations} permutations. Z maps were FDR-corrected for significance at alpha = {alpha}. The analysis was completed within {time_taken}. </p>

            </div>

            <div class="section highlight">
                <h3>Key Parameters</h3>
                <ul>
                    <li><strong>Number of patients:</strong> {num_patients}</li>
                    <li><strong>Number of permutations:</strong> {n_permutations}</li>
                    <li><strong>Alpha level:</strong> {alpha}</li>
                </ul>
            </div>

            
            <div class="section">
                <h2>Lesion Overlap</h2>
                    <h3>Unfiltered<h3>
                    <img src = "lesion_overlap_mosaic.png" alt = "lesion overlap" style="width: 100%; height: auto;">
                
                """)
        if min_patient_count>0:
            f.write(f"""
                    <br><br>
                
                    <h3>Filtered (Minimum {min_patient_count} patients)</h3>
                    <img src = "lesion_overlap_filtered_mosaic.png" alt = "lesion overlap filtered by {min_patient_count} patients" style="width: 100%; height: auto;">
                """)
        f.write(f"""
            </div>
            
            <div class="section">
                <h2>SVR Parameters</h2>
                <table class="result-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
            """)
        for param, value in svr_params.items():
            f.write(f"<tr><td>{param}</td><td>{value}</td></tr>\n")

        f.write(f"""
                </table>
                <h3>Unthresholded SVR-Beta map</h3>
                <img src = "svr_beta_map_mosaic.png" alt = "Unthresholded beta map" style="width: 100%; height: auto;">
                
                <h3>Permutation Tested<h3>
                <h3>Unthresholded SVR Z-Map</h3>
                <img src = "zmap_mosaic.png" alt = "Permutation tested zmap" style="width: 100%; height: auto;">
                
                <h3>p<0.05 SVR Z-Map</h3>
                <img src = "zmap_p05_mosaic.png" alt = "Permutation tested zmap" style="width: 100%; height: auto;">
                
                <h3>p<0.01 SVR Z-Map</h3>
                <img src = "zmap_p01_mosaic.png" alt = "Permutation tested zmap" style="width: 100%; height: auto;">
                
                <h3>p<0.005 SVR Z-Map</h3>
                <img src = "zmap_p005_mosaic.png" alt = "Permutation tested zmap" style="width: 100%; height: auto;">
                
                <h3>p<0.001 SVR Z-Map</h3>
                <img src = "zmap_p001_mosaic.png" alt = "Permutation tested zmap" style="width: 100%; height: auto;">
            </div>
            
            <div class="section highlight">
                <h3>Results</h3>
                <ul>
                    <li><strong>Mean lesion volume (in voxels):</strong> {mean_lesion_volume:.2f}</li>
                    <li><strong>Number of lesion files:</strong> {num_lesions}</li>
                    <li><strong>Z-map value range:</strong> ({zmap_range[0]:.2f}, {zmap_range[1]:.2f})</li>
                </ul>
            </div>
            <div id="section">
                <h2>Interactive Viewer</h2>
    
                <!-- Embedding another HTML file using iframe -->
                {html_view}
    
            </div>

        """)


        f.write(f"""
                <div class="section">
                    <h3>Download</h3>
                    <p>Download the <a href='zmap.nii.gz'>Z-map NIfTI file</a> for further analysis.</p>
                </div>
            """)

        if fileExists:
            f.write(f"""
                <div class="section">
                <h2>Overview</h2>
                <img src="data:image/png;base64,{overview_image_base64}" alt="Overview Brain Image" style="width: 100%; height: auto;">


                <h2>Cluster Details</h2>

                """)
            for idx, row in top_clusters.iterrows():
                f.write(f"""
                <div class="section highlight" style="margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
                    <img src="data:image/png;base64,{cluster_images_base64[idx]}" style="width: 100%; height: auto; display: block;" alt="Cluster Image">
                    <p style="font-size: 16px; line-height: 1.6; padding-top: 10px;">
                        <strong>Coordinates (X, Y, Z):</strong> ({row['peak_x']}, {row['peak_y']}, {row['peak_z']})<br>
                        <strong>Cluster Mean:</strong> {row['cluster_mean']:.2f}<br>
                        <strong>Volume (mm³):</strong> {row['volume_mm']:.2f}<br>
                        <strong>Desikan-Killiany:</strong><br> {row['desikan_killiany'].replace("_", " ").replace(";", "<br>")}<br>
                        <strong>Harvard-Oxford:</strong><br> {row['harvard_oxford'].replace("_", " ").replace(";", "<br>")}
                    </p>
                </div>
                """)
        else:
            f.write("""
                <div class="section">
                    <p>No significant clusters found</p>
                </div>
            """)
        f.write("""</table></div>

            <div class="footer">
                <p>Generated by SVR-Based Lesion-Symptom Mapping Tool</p>
            </div>

            <!-- Lightbox script -->
            <script>
                function openLightbox(event, element) {
                    event.preventDefault();  // Prevent the default anchor click action
                    var lightbox = document.createElement('div');
                    lightbox.classList.add('lightbox');
                    var img = document.createElement('img');
                    img.src = element.href;
                    lightbox.appendChild(img);
                    document.body.appendChild(lightbox);

                    // Add an event listener to close the lightbox when clicked
                    lightbox.addEventListener('click', function() {
                        lightbox.remove();
                    });
                }
            </script>
            </div>
        </body>
        </html>
        """)

    print(f"Report successfully saved to {output_file}.")


