from atlasreader import create_output
def atlas_read_cluster(file, output_dir, cluster_extent=30):
    create_output(file, cluster_extent=cluster_extent, outdir=output_dir)
    print("Atlas reader clusters saved in:\n", output_dir)