from em_methods.fdtd.fdtd import fdtd_run
from em_methods.optimization.pso import particle_swarm

fdtd_file: str = "D:\\Ivan S\\BURST\\c-Si\\tests\\voids_c-Si.fsp"

param_dict = {
    "a_i": [0.5, 1],
    "z_radius_i": [0.3e-6, 0.5e-6],
    "xy_radius_i": [0.3e-6, 0.5e-6],
    "fARC_i": [10e-9, 50e-9],
}

def fdtd_func(a_i, z_radius_i, xy_radius_i, fARC_i):
    FoM_matrix = []
    generator = enumerate(zip(a_i, z_radius_i, xy_radius_i, fARC_i))
    for index, (a_i, z_radius_i, xy_radius_i, fARC_i) in generator:
        print(
            f"index, fARC, ax, xy_radius, z_radius = {index}, {fARC_i:.4g}, {a_i:.4g}, {xy_radius_i:.4g}, {z_radius_i:.4g}."
        )
        properties = {
            "Domes": {
                "ax": a_i,
                "ay": a_i,
                "z_radius": z_radius_i,
                "radius": xy_radius_i,
            },
            "fARC": {"z span": fARC_i},
        }
        results = {"results": {"solar_generation": ["Jsc"]}}
        prefix: str = (
            f"{index}_a_i{a_i:.3g}_z_radius{z_radius_i:.3g}_radius{xy_radius_i:.3g}_fARC{fARC_i:.3g}"
        )
        res, *_ = fdtd_run(
            fdtd_file, properties, results, override_prefix=prefix, savepath="D:\\Ivan S\\BURST\\c-Si\\tests\\data"
        )

        jsc = res["results.solar_generation.Jsc"]
        FoM_matrix.append(jsc)

        # check Jsc
        if jsc == 0 or jsc == np.nan:
            print(f"Jsc was not determined on the th cycle.:")
            break
        else:
            print(f"Jsc of th cycle is equal to {jsc:.4g}.")

        # # single values
        # df.loc[index] = [fARC_i, a_i, xy_radius_i, z_radius_i, jsc]
        # current_df = pd.read_parquet(pq_file)
        # full_df = current_df.merge(df, how="outer")
        # full_df.to_parquet(path=pq_file)

        if os.path.exists(f"D:\\Ivan S\\BURST\\c-Si\\tests\\data\\{prefix}_voids_c-Si.fsp"):
            os.remove(f"D:\\Ivan S\\BURST\\c-Si\\tests\\data\\{prefix}_voids_c-Si.fsp")
            print("The FDTD file has been removed successfully.")
        else:
            print("The FDTD file does not exist.")

    return FoM_matrix

import logging

logger = logging.getLogger("sim")
logger.setLevel(logging.INFO)

particle_swarm(
    fdtd_func,
    param_dict,
    maximize=True,
    # pso_gui=False,
    inert_prop=(0.9, 0.4, True),
    ind_cog=1.39,
    soc_learning=1.39,
    particles=25,
    iterations=(10, 15, True),
    tolerance=(0.05, 10),
    progress=True,
    export=True,
    basepath="D:\\Ivan S\\BURST\\c-Si\\tests\\data",
)