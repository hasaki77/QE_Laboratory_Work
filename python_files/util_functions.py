import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from plotly import express as px
import numpy as np
from scipy import*
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from ase.io import write, read
import subprocess
import yaml
from yaml import CLoader as Loader
import pickle
from munch import Munch

from ase.io.vasp import read_vasp_out, write_vasp, read_vasp_xml
from ase.io import write, read
from ase.build import make_supercell, graphene
from ase import Atoms, Atom
from ase.io.lammpsdata import read_lammps_data
from ase.geometry import cell_to_cellpar
from pathlib import Path, PurePath

def PhononDisp(X, Y, freq_unit: str, savedir: str, tick_vals, tick_text, title: str, method='direct', save=False, show=False) -> None:
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    color_dict = px.colors.qualitative.Plotly
    # GPUMD
    if method == 'loop':
        if freq_unit == 'thz':
            freq_unit = 'THz'
            for i, y_val in enumerate(Y):
                fig.add_trace(
                    go.Scatter(x=X,
                            y=y_val,
                            line_color=color_dict[0],
                            line_width=5,
                            showlegend=False,
                            name=''
                            ),
                    secondary_y=False,)
        elif freq_unit == 'cm-1':
            freq_unit = f'cm\u207B\N{SUPERSCRIPT ONE}'
            for i, y_val in enumerate(Y):
                fig.add_trace(
                    go.Scatter(x=X,
                            y=y_val*33.35641,
                            line_color=color_dict[0],
                            line_width=5,
                            showlegend=False,
                            name=''
                            ),
                    secondary_y=False,)
    # PHONOPY
    elif method == 'direct':
        if freq_unit == 'thz':
            freq_unit = 'THz'
            fig.add_trace(
                go.Scatter(x=X,
                        y=Y,
                        line_color=color_dict[0],
                        line_width=5,
                        showlegend=False,
                        name=''
                        ),
                secondary_y=False,) 
        elif freq_unit == 'cm-1':
            freq_unit = f'cm\u207B\N{SUPERSCRIPT ONE}'
            fig.add_trace(
                go.Scatter(x=X,
                        y=Y*33.35641,
                        line_color=color_dict[0],
                        line_width=5,
                        showlegend=False,
                        name=''
                        ),
                secondary_y=False,)               
    # Add figure title
    fig.update_layout(font_size = 50,
                    font_color='black',
                    title = f"{title}",
                    title_y = 0.99,
                    title_x = 0.5,
                    #legend_title = "Measurement<br>parameters",
                    legend_font_size = 50,
                    plot_bgcolor = 'rgba(250,250,250,1)',
                    width = 1000,
                    height = 600,   
                    # X-axis
                    xaxis_title = "K-path",
                    #xaxis_nticks = 7,
                    xaxis_ticklen = 16,
                    xaxis_tickwidth = 3,
                    xaxis_ticks = 'outside',
                    #xaxis_range = [0, 5000],
                    xaxis_tickvals = tick_vals,
                    xaxis_ticktext = tick_text,
                    # Y-axis-right
                    yaxis_title = f"Frequency, {freq_unit}",
                    yaxis_titlefont_color=color_dict[0],
                    yaxis_nticks = 12,
                    yaxis_ticklen = 16,
                    yaxis_tickwidth = 3,                                           
                    yaxis_ticks = 'outside',
                    # Y-axis-left
                    yaxis2_title = "Refractive Index",
                    yaxis2_titlefont_color=color_dict[2],
                    yaxis2_nticks = 12,
                    yaxis2_ticklen = 16,
                    yaxis2_tickwidth = 3,                                           
                    yaxis2_ticks = 'outside' 
                    )
    for i in tick_vals:
        fig.add_vline(i)
    
    fig.add_shape(type="rect",
                    xref="paper",
                    yref="paper",
                    x0=0,
                    y0=0,
                    x1=0.94,
                    y1=1.0,
            line=dict(
                color="black",
                    width=3,))
    fig.layout.font.family = 'sans-serif'
    if save:
        fig.write_image(file=savedir  / 'phonon_band.png',
                        format='png',
                        width=2250,
                        height=1500,
                        scale=1
                        )
        # Save coordinate points
        with (savedir / 'phonon_band_points.pkl').open(mode='wb') as f:
            pickle.dump(Munch(x=X, y=Y), f)
            f.close()
    if show:
        fig.show()

def PhDoS_matplotlib(X, Y, freq_unit: str, savedir: str, title: str, save=False, show=False) -> None:
    plt.style.use('seaborn-v0_8-paper')

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    cm_freq = X *33.35641

    ax.plot(cm_freq, Y, label='', color='blue', linewidth=.5)

    ax.set_title(title, fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel(f"Frequency, {freq_unit}", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("DOS", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim([cm_freq.min(), cm_freq.max()])
    ax.set_ylim([Y.min(), Y.max()])

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=5)

    ax.legend(loc="upper right", fontsize=10, frameon=False)
    if save:
        plt.savefig(savedir  / 'phdos.png', dpi=600, bbox_inches='tight')
    if show:
        plt.show()

    # FIGURE FOR COMBINATION WITH BAND STRUCTURE
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.plot(-cm_freq, Y, label='', color='blue', linewidth=.5)
    ax.set_title(title, fontsize=16, fontweight='bold', color='black')
    ax.set_xticks([])
    ax.set_ylabel("", fontsize=12, fontweight='bold', labelpad=10)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlim([-cm_freq.max(), -cm_freq.min()])
    ax.set_ylim([Y.min(), Y.max()])
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=0, labelrotation=90, labelcolor='white')
    if save:
        plt.savefig(savedir  / 'phdos_for_comb.png', dpi=600, bbox_inches='tight')


def PhononDisp_matplotlib(X, Y, freq_unit: str, savedir: str, xtick_vals, xtick_text, title: str, save=False, show=False) -> None:
    plt.style.use('seaborn-v0_8-paper')

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    cm_freq = Y *33.35641

    ax.plot(X, cm_freq, label='', color='blue', linewidth=.5)

    ax.set_title(title, fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel("", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(f"Frequency, {freq_unit}", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([cm_freq.min(), cm_freq.max()])

    ax.vlines(xtick_vals, ymin=cm_freq.min(), ymax=cm_freq.max(), color='black', linewidth=0.5)
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_text)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=5)

    ax.legend(loc="upper right", fontsize=10, frameon=False)
    if save:
        plt.savefig(savedir  / 'phonon_band.png', dpi=600, bbox_inches='tight')
    if show:
        plt.show()

def create_lammps_in(
                file_path: str,
                nep_path: str,
                atom_types: list,
                atom_masses: list,
                forcefile: str='dump.force',
                units: str='metal',
                dimension: int=3,
                boundary: str='p p p',
                atom_style: str='atomic',
                read_data: str='supercell',
                pair_style: str='nep',
                thermo_step: int=100,
                run_step: int=0
):
        f = open(file_path, 'w')
        f.write(f'units          {units}\n')
        f.write(f'dimension      {dimension}\n')
        f.write(f'boundary       {boundary}\n')
        f.write(f'atom_style     {atom_style}\n')
        f.write(f'read_data      {read_data}\n')
        f.write('\n')

        f.write(f'pair_style     {pair_style}\n')
        f.write(f'pair_coeff * * {nep_path} {" ".join(atom_types)}\n')
        f.write('\n')

        for i, mass in enumerate(atom_masses):
                f.write(f'mass {i+1} {mass}\n')
        f.write('\n')
        
        #f.write(f'compute myforce all property/atom fx fy fz\n')
        f.write(f'dump phonopy all custom 1 {forcefile} id type x y z fx fy fz\n')
        f.write(f'dump_modify phonopy format line "%d %d %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f"\n')
        f.write(f'run {run_step}\n')
        f.close()

def create_forces(force_dir, supercell_dim):
        with open("FORCE_SETS", "w") as outfile:
                outfile.write(f"{supercell_dim}\n")
                outfile.write(f'24\n24\n\n')
                for forcefile in force_dir:               
                        with open(forcefile, "r") as infile:
                                forces = []
                                for line in infile:
                                        parts = line.split()
                                        if len(parts) == 8:
                                                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                                                forces.append((fx, fy, fz))

                                #outfile.write(f"# File {forcefile}\n")
                                outfile.write("1\n")
                                for force in forces:
                                        outfile.write(f"{force[0]} {force[1]} {force[2]}\n")

                                outfile.write("\n")  # Силы для текущей структуры

def match_then_insert(filename, match, content):
    lines = open(filename).read().splitlines()
    index = lines.index(match)
    lines.insert(index, content)
    open(filename, mode='w').write('\n'.join(lines))

def check_running_jobs():
    result = subprocess.run(["squeue"], capture_output=True, text=True)
    if result.returncode == 0:
        # Jobs count
        job_count = len(result.stdout.strip().split("\n")) - 1
        print(f"Number of run jobs: {job_count}")
        return job_count
    else:
        print("Error of task checking", result.stderr)
        return 0

def compute_overlap(local_modes, normal_modes):
    real_part = 0.0
    imag_part = 0.0
    for i, lmode in enumerate(local_modes):
        nmode = normal_modes[i]
        for j, ldim in enumerate(lmode):
            ndim = nmode[j]
            real_part += ldim[0]*ndim[0] + ldim[1]*ndim[1]
            imag_part += ldim[1]*ndim[0] - ldim[0]*ndim[1]
    overlap = np.sqrt(real_part*real_part + imag_part*imag_part)
    return overlap


def reordering(local_freqs, local_modes, normal_freqs, normal_modes):
    outvec = []
    outmode = []
    index_list = []
    for j, nmod in enumerate(normal_modes):
        max_index = j
        max_overlap = 0.0
        for k, lmod in enumerate(local_modes):
            if (k in index_list or np.fabs(normal_freqs[j] - local_freqs[k]) > 0.5):
                continue
            overlap = compute_overlap(lmod, nmod)
            if (overlap - max_overlap) > 1.0E-6:
                max_index = k
                max_overlap = overlap
        index_list.append(max_index)
        outvec.append(local_freqs[max_index])
        outmode.append(local_modes[max_index])
    return (outvec, outmode)

def read_band_yaml(filename):

    with open('band.yaml') as f:
        read_data = yaml.safe_load(f)

    distances = []
    frequencies = []
    data = read_data.get('phonon', {})
    for point in data:
        distances.append(point["distance"])
        point_frequencies = [band["frequency"] for band in point["band"]]
        frequencies.append(point_frequencies)
        
    segment_nqpoint = read_data.get('segment_nqpoint', {})
    end_points = np.zeros(len(segment_nqpoint)+1, dtype=np.int16)
    for i, nq in enumerate(segment_nqpoint):
        end_points[i+1] = nq + end_points[i]
    end_points[-1] -= 1
    segment_positions = np.array(distances)[end_points]
    
    return np.array(distances), np.array(frequencies), np.array(segment_nqpoint), segment_positions

def make_conf_files(
        work_dir: str,
        atom_name: str, 
        supercell: str,
        displacement: float,
        write_disp: str,
        symprec: float,
        fc_symmetry: str,
        mesh_pts: str,
        tmax: int,
        tmin: int,
        tstep: int,
        band: str,
        band_points: int,
        band_connection: str,
        eigenvectors: str,
):
    # PHONOPY.CONF
    with (work_dir / 'phonopy.conf').open(mode='w') as f:
        f.write(f'ATOM_NAME = {atom_name}\n')
        f.write(f'DIM = {supercell}\n')
        f.write(f'DISPLACEMENT_DISTANCE = {displacement}\n')
        f.write(f'WRITE_DISPLACEMENTS = {write_disp}\n')
        #f.write(f'PRIMITIVE_AXIS = AUTO\n')
        f.write(f'SYMPREC = {symprec}\n')
        f.write(f'FC_SYMMETRY = {fc_symmetry}\n')
        f.close()

    # MESH.CONF
    with (work_dir / 'mesh.conf').open(mode='w') as f:
        f.write(f'ATOM_NAME = {atom_name}\n')
        f.write(f'DIM = {supercell}\n')
        f.write(f'MP = {mesh_pts}\n')
        f.close()

    # MESH_THERM_PROP.CONF
    with (work_dir / 'mesh_therm_prop.conf').open(mode='w') as f:
        f.write(f'ATOM_NAME = {atom_name}\n')
        f.write(f'DIM = {supercell}\n')
        f.write(f'MP = {mesh_pts}\n')
        f.write(f'TMAX = {tmax}\n')
        f.write(f'TMIN = {tmin}\n')
        f.write(f'TSTEP = {tstep}\n')
        f.close()
    
    # BAND.CONF
    with (work_dir / 'band.conf').open(mode='w') as f:
        f.write(f'ATOM_NAME = {atom_name}\n')
        f.write(f'DIM = {supercell}\n')
        #f.write(f'PRIMITIVE_AXIS = AUTO\n')
        f.write(f'BAND = {band}\n')
        f.write(f'BAND_POINTS = {band_points}\n')
        f.write(f'BAND_CONNECTION = {band_connection}\n')
        f.write(f'EIGENVECTORS = {eigenvectors}\n')
        f.close()

def shift_structure(main_dir, take_from: str, save_into: str, shift_structure: str, shift_coef: np.array):

    structure = read(main_dir / take_from / shift_structure)
    cell_params = cell_to_cellpar(structure.cell)
    shift_vector = np.array(cell_params[:3]) * shift_coef
    structure.translate(shift_vector)
    structure.wrap()
    # Write new file
    write(main_dir / save_into / 'shifted_structure.vasp', structure)
    # # Remove old file
    # os.remove(main_dir / take_from / shift_structure)
    # Rename new file
    Path(main_dir / save_into / 'shifted_structure.vasp').rename(main_dir / save_into / shift_structure)

def create_kpoints(work_dir: str, kpts: str):
    with (work_dir / 'KPOINTS').open(mode='w') as f:
        f.write('Automatic generation\n')
        f.write('0\n')
        f.write(f'Monkhorst-pack\n')
        f.write(f'{kpts}\n')
        f.write('0  0  0\n')

def create_folder(main_dir: str, folder: str):
    folder_path = main_dir / folder
    if not folder_path.exists():
        folder_path.mkdir()
        print(f'Folder {folder_path.name} is created')
    else:
        print(f'Folder {folder_path.name} already exists')
    
    return folder_path

def save_path(main_dir: Path, dir_name: str, general_paths_dir: Path, munch_dict: Munch, general_dict: Munch):
    munch_dict[dir_name] = munch_dict.setdefault(dir_name, Munch())
    munch_dict[dir_name].__val__ = main_dir
    with (general_paths_dir).open(mode='w') as f:
        yaml.dump(munch_to_dict(general_dict), f)
        f.close()
    print('File path is saved')

def munch_to_dict(obj):
    if isinstance(obj, Munch):
        d = {}
        for k, v in obj.items():
            d[k] = munch_to_dict(v)
        return d
    elif isinstance(obj, list):
        return [munch_to_dict(i) for i in obj]
    else:
        return obj

def dict_to_munch(d):
    if isinstance(d, dict):
        return Munch({k: dict_to_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_munch(i) for i in d]
    else:
        return d

def create_txt(filename):
    ''' 
    'r' (read-only) (default), 'w' (write - erase all contents for existing file), 'a' (append), 'r+' (read and write)
    '''
    with open(filename, "w") as file:
        file.close()
        
def rewrite_file(work_dir, root_file: str, folder_name: str, file_type: str, new_line: bool | dict) -> None:
    # Create folder
    folder_path = work_dir / folder_name
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Create txt file
    step_file = folder_path / file_type
    create_txt(step_file)

    # Rewrite txt file with changed parameter values
    if new_line:
        with open(root_file, 'r+') as f1, open(step_file, 'w') as f2:
            for line1 in f1:
                for fr in new_line.keys():
                    if fr in line1:
                        f2.write(new_line[fr])
                    else:
                        f2.write(line1)
            f1.close(), f2.close()
    else:
        with open(root_file, 'r+') as f1, open(step_file, 'w') as f2:
            for line1 in f1:
                f2.write(line1)
            f1.close(), f2.close()

def create_bash_script(
    script_filename: str,
    job_name: str,
    partition: str,
    nodes: int,
    memory: str,
    exclude: str,
    ntasks: int,
    cpus_per_task: int,
    gpus: int,
    time: int,
    command: str

): 
    with script_filename.open(mode='w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --partition={partition}\n')
        f.write(f'#SBATCH --nodes={nodes}\n')
        f.write(f'#SBATCH --mem-per-cpu={memory}\n')
        if exclude:
            f.write(f'#SBATCH --exclude={exclude}\n')
        else:
            f.write(f'##SBATCH --exclude={exclude}\n')
        f.write(f'#SBATCH --ntasks={ntasks}\n')
        f.write(f'#SBATCH --cpus-per-task={cpus_per_task}\n')
        if gpus:
             f.write(f'#SBATCH --gpus={gpus}\n')
        else:
             f.write(f'##SBATCH --gpus={gpus}\n')
        f.write(f'#SBATCH --time={time}:00:00\n\n')

        f.write('ulimit -c unlimited\n')
        f.write('ulimit -s unlimited\n\n')

        f.write('#COMMAND PART\n')
        f.write(f'{command}\n')

def create_dos_in(
    main_dir: str,
    prefix: str,
    deltaE: float,
):
    with main_dir.open(mode='w') as f:
        f.write('&dos\n')
        f.write(f'prefix={prefix}\n')
        f.write(f"outdir = '.'\n")
        f.write(f'DeltaE = {deltaE}\n')
        f.write('/\n')

def create_bands_in(
    main_dir: str,
    prefix: str,
): 
    with main_dir.open(mode='w') as f:
        f.write('&bands\n')
        f.write(f'prefix={prefix}\n')
        f.write(f"outdir = '.'\n")
        f.write('/\n')
        
def cut_layers(filename: Path, coord: tuple, num_sections: int, P_cell: list):
    # Read Structure
    structure = read(filename)
    # Make Supercell
    super_structure = make_supercell(structure, P_cell)
    # Cut axis to pieces
    axis_coord = super_structure.get_positions()[:, coord]
    piece = ( axis_coord.max() - axis_coord.min() ) / num_sections

    # Get Layers
    layers = []
    for section in range(num_sections):
        if section == num_sections - 1:
            layer_index = np.where( (axis_coord >= axis_coord.min()+(section*piece)) & (axis_coord <= axis_coord.min() + (section+1)*piece) )[0]
        else:
            layer_index = np.where( (axis_coord.min()+(section*piece) <= axis_coord) & (axis_coord < axis_coord.min() + (section+1)*piece) )[0]
        layers.append(super_structure[layer_index])
    
    return layers
    
def get_vertices(a, b, c):
    vertices = np.array([
        [0, 0, 0],  # вершина 1 (0, 0, 0)
        a,  # вершина 2 (a, 0, 0)
        b,  # вершина 3 (0, b, 0)
        c,  # вершина 4 (0, 0, c)
        a + b,  # вершина 5 (a, b, 0)
        a + c,  # вершина 6 (a, 0, c)
        b + c,  # вершина 7 (0, b, c)
        a + b + c  # вершина 8 (a, b, c)
    ])
    return vertices

def shift_cell_volume(vertices, Lcell: float, coord: int):

    refer_vert = 0
    edge_vert = [1, 2, 3][coord]
    edge_indices_list = [
        [1, 4, 5, 7],
        [2, 4, 6, 7],
        [3, 5, 6, 7]
    ]
    edge_indices = edge_indices_list[coord]

    direction_vector = vertices[edge_vert] - vertices[refer_vert] 
    
    current_length = np.linalg.norm(direction_vector)
    # Вычисляем коэффициент масштабирования
    scale_factor = Lcell / current_length
    # Новый вектор с нужной длиной
    shift_vector = direction_vector * (scale_factor - 1)
    # Обновляем координаты вершин, сдвигая их
    vertices[edge_indices] += shift_vector
    
    return vertices

def rearrange_atoms_poscar(layers,):
    comb_struc = []
    for symbol in set(layers.get_chemical_symbols()):
        comb_struc += [atom for atom in layers if atom.symbol == symbol]

    comb_struc = Atoms(symbols=[atom.symbol for atom in comb_struc],
        positions=[atom.position for atom in comb_struc],
        cell=layers.get_cell(),
        pbc=layers.get_pbc(),
        celldisp=layers.get_celldisp())
    return comb_struc

def rotate_atoms(positions, angle: float, axis: int, center: str):
    
    # Center of masses
    if center is None:
        center_point = np.mean(positions, axis=0)
    elif isinstance(center, str):
        if center is 'COM':
            center_point = np.mean(positions, axis=0)

        elif center is '0AXIS':
            center_point = np.array([0.0, 0.0, 0.0])
        else:
            raise ValueError('Unknown center specifier')
    else:
        center_point = np.array(center)

    # Transfer atoms to the center of rotations
    shifted_positions = positions - center_point        
    # Angles
    theta = np.radians(angle)

    # Choose rotation matrix for the axis
    if axis == 0:
        rotation_matrix = np.array([[1, 0, 0],
                                    [0,  np.cos(theta), -np.sin(theta)],
                                    [0,  np.sin(theta),  np.cos(theta)]])
    elif axis == 1:
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 2:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Please mention the axis: x: 0, y: 1, z: 2")

    # Rotate Atoms
    rotated_positions = shifted_positions @ rotation_matrix.T + center_point

    return rotated_positions

def translate_points(points, direction_vect, length):
    shift_vector = length * direction_vect
    points += shift_vector
    return points

def make_symmetric_supercell(positions, cell, size):
    offsets = - np.array([size[i]//2 * cell[i] for i in range(len(size))]).sum(axis=0)
    return positions + offsets

def cut_atoms_inside_cell(positions, cell, epsilon=1e-6):
    atoms_in_cell = []
    symbols_idx = []
    # Matrix of edge cell
    M = np.array([
        cell[0],
        cell[1],
        cell[2]
    ]).T

    for idx, position in enumerate(positions):
        # Solve system of equations
        coeffs = np.dot(np.linalg.inv(M), position)

        if np.allclose(coeffs, np.clip(coeffs, -epsilon, 1 - epsilon)):
            atoms_in_cell.append(position)
            symbols_idx.append(idx)
    
    return np.array(atoms_in_cell)

def find_degrees(m_arr: np.arange, n_arr: np.arange, dim=2):
    degrees = dict()
    for m, n in np.array(np.meshgrid(m_arr, n_arr)).T.reshape(-1, dim):
        a = n**2 + 4*n*m + m**2 
        b = 2*(n**2 + n*m + m**2)                    
        degree = np.round(np.degrees(np.arccos(a / b)), 3)
        num_atoms = 4*(n**2 + n*m + m**2)
        
        if degree not in degrees.keys():
            degrees[degree] = (n, m)
    
    return degrees

def create_lammps_relax(
                file_path: str,
                nep_path: str,
                atom_types: list,
                atom_masses: list,
                read_data: str,
                isif: str,
                ibrion: str,
                potim: float,
                ediff: float,
                ediffg: float,
                steps: int,
                cutoff: float,
                units: str='metal',
                dimension: int=3,
                boundary: str='p p p',
                atom_style: str='atomic',
                pair_style: str='nep',
                thermo_step: int=100,
                run_step: int=0
):
        f = file_path.open(mode='w')
        f.write(f'units          {units}\n')
        f.write(f'dimension      {dimension}\n')
        f.write(f'boundary       {boundary}\n')
        f.write(f'atom_style     {atom_style}\n')
        f.write(f'read_data      {read_data}\n')
        f.write('\n')

        f.write(f'pair_style     {pair_style}\n')
        f.write(f'pair_coeff * * {nep_path} {atom_types}\n')
        f.write('\n')

        for i, mass in enumerate(atom_masses):
                f.write(f'mass {i+1} {mass}\n')
        f.write('\n')

        f.write(f"neighbor {cutoff} bin\n")
        f.write(f"neigh_modify delay 0 every 1 check yes\n")
        f.write(f"thermo {thermo_step}\n")
        f.write(f"thermo_style custom step pe etotal press lx ly lz\n\n")

        f.write(f'{isif}\n')
        f.write(f'min_style {ibrion}\n')
        f.write(f'min_modify dmax {potim}\n')
        f.write(f'minimize {ediff} {ediffg} {steps}\n\n')


        f.write(f'dump 1 all custom 100 relaxed.xyz id type x y z fx fy fz\n') 
        f.write(f'write_data relaxed_structure.data\n')
        f.close()