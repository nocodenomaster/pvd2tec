import sys
from io import StringIO
import numpy as np


def xml_line_decoder(line):
    # clean #0
    id0 = line.index('<')
    id1 = line.index('>')
    clean0 = line[id0:id1].split(' ')
    # clean #1
    array_dict = {}
    for section in clean0:
        if section.find('=') == -1:
            continue
        else:
            pair = section.split('=')
            pair[1] = pair[1].replace("\"", "")
            array_dict[pair[0]] = pair[1]
    return array_dict


class PVDFile:
    def __init__(self, filename):
        self._quantity_name = ""
        self._file = []
        self._mesh_dim = -1
        self._quantity_dim = -1
        self.get_file_info(filename)

    def get_file_info(self, pvd_name):
        split_id = pvd_name.rfind('/')
        pathname, filename = pvd_name[:split_id + 1], pvd_name[split_id + 1:]
        # open pvd to find pvtu
        isfind = 0
        with open(pvd_name, 'r') as fpvd:
            content_fpvd = fpvd.readlines()
        for line in content_fpvd:
            if line.find('file') != -1:
                clean_dict = xml_line_decoder(line)
                pvtu_name = clean_dict['file']
                isfind = 1
                break
        else:
            pvtu_name = ''
            ValueError('Cannot find *.pvtu')
        if isfind == 0:
            raise RuntimeError('can\'t find *.pvtu')
        # open pvtu to find vtu
        with open(pathname + pvtu_name, 'r') as fpvtu:
            content_fpvtu = fpvtu.readlines()
            if any('Scalars' in line for line in content_fpvtu):
                quantity_type = 'Scalars'
                self._quantity_dim = 1
            elif any('Vectors' in line for line in content_fpvtu):
                quantity_type = 'Vectors'
                self._quantity_dim = 3
            else:
                raise RuntimeError('Unknown quantity type, not scalar and vector.')
            # quantity name
            for line in content_fpvtu:
                name_id = line.find(quantity_type)
                src_id = line.find('Source')
                arg = xml_line_decoder(line)
                if len(arg) == 0:
                    continue

                if 'NumberOfComponents' in arg and len(arg) == 2:
                    self._mesh_dim = int(arg['NumberOfComponents'])
                if name_id >= 0:
                    self._quantity_name = arg[quantity_type]
                elif src_id >= 0:
                    self._file.append(pathname + arg['Source'])

    def read_vtu(self, vtuname, mode='default'):
        # hard code, assume tetrahedron
        gdim = 3
        nvtx = 0
        if gdim == 3:
            nvtx = 4
        elif gdim == 2:
            nvtx = 3
        elif gdim == 1:
            nvtx = 2

        if mode.find('default') != -1:
            mode = 'pvc'
        elif mode.find('mesh') != -1:
            mode = 'pc'
        elif mode.find('append') != -1:
            mode = 'pv'

        with open(vtuname, 'r') as fvtu:
            context_fvtu = fvtu.readlines()
            if 'p' in mode:
                point_str = context_fvtu[5].replace('>', '<').split('<')[2].split(' ')
                newpoint = np.array([float(p) for p in point_str if p != '']).reshape(-1, gdim)
            else:
                newpoint = None
            if 'v' in mode:
                value_str = context_fvtu[13].replace('>', '<').split('<')[2].split(' ')
                newvalue = np.array([float(v) for v in value_str if v != '']).reshape(-1, self._quantity_dim)
            else:
                newvalue = None
            if 'c' in mode:
                connectivity_str = context_fvtu[8].replace('>', '<').split('<')[2].split(' ')
                newconnectivity = np.array([int(c) for c in connectivity_str if c != '']).reshape(-1, nvtx)
            else:
                newconnectivity = None
        return newpoint, newvalue, newconnectivity

    def get_quantity_name(self):
        dim = self._quantity_dim

        # when the quantity is a vector
        if dim > 1:
            name = self._quantity_name
            # name of vector components
            namelist = [name + str(i) for i in range(dim)]
            # name of vector norm
            namelist.append('|' + name + '|')

            return namelist
        return [self._quantity_name]

    def get_quantity_dim(self):
        return self._quantity_dim

    def get_files(self):
        return self._file


class PVDMesh(PVDFile):
    def __init__(self, filename):
        super().__init__(filename)
        self.custom_setup()

    def custom_setup(self):
        self._filelist = self.get_files()
        point = []
        connectivity = []
        for filename in self._filelist:
            newpoint, _, newconnectivity = self.read_vtu(filename, 'mesh')
            if len(connectivity) == 0:
                connectivity = newconnectivity
            else:
                connectivity = np.vstack((connectivity, newconnectivity + len(point)))
            if len(point) == 0:
                point = newpoint
            else:
                point = np.vstack((point, newpoint))
        pset = {}
        cset = []
        k = 0
        for i in range(point.shape[0]):
            p = point[i]
            if tuple(p) in pset:
                continue
            else:
                pset[tuple(p)] = k
                k += 1
        for i, c in enumerate(connectivity):
            cset += [[pset[tuple(point[idx])] for idx in c]]
        self._point = np.array([np.array([*key]) for key, value in pset.items()])
        self._connectivity = np.array(cset)
        self._vtx2dof = pset

    def get_points(self):
        return self._point

    def get_elements(self):
        return self._connectivity

    def get_num_points(self):
        return self._point.shape[0]

    def get_num_elements(self):
        return self._connectivity.shape[0]


class PVDData:
    def __init__(self, mesh):
        self._mesh = mesh

    def read_pvd(self, filename):
        f = PVDFile(filename)
        vdim = f.get_quantity_dim()
        vnum = self._mesh.get_num_points()
        self._data = np.zeros((vnum * vdim,))
        self._name = f.get_quantity_name()
        self._dim = vdim
        for filename in f.get_files():
            points, values, _ = f.read_vtu(filename, 'append')
            for i in range(points.shape[0]):
                p = tuple(points[i])
                idx = self._mesh._vtx2dof[p]
                self._data[idx * vdim: idx * vdim + vdim] = values[i, :]
        self._data = self._data.reshape((-1, vdim))

    def get_data(self):
        data = self._data
        # get norm
        if data.shape[1] > 1:
            data_norm = np.linalg.norm(data, 2, axis=1).reshape((-1, 1))
            return np.hstack((data, data_norm))
        return data

    def get_name(self):
        return self._name

    def get_dim(self):
        return self._dim


def writeTEC(tecname, mesh, datalist):
    string = StringIO()
    x = mesh.get_points()
    name = []
    if x.shape[1] == 3:
        name = ['X', 'Y', 'Z']
    elif x.shape[1] == 2:
        name = ['X', 'Y']
    elif x.shape[1] == 1:
        name = ['X']
    connectivity = mesh.get_elements()
    if isinstance(datalist, PVDData):
        datalist = [datalist]
    for ref in datalist:
        if not isinstance(ref, PVDData):
            raise TypeError(f"ref ({type(ref)}) is not PVDData")
        dim = ref.get_dim()
        for newname in ref.get_name():
            name.append(newname)
        x = np.hstack((x, ref.get_data()))
    varname = "VARIABLES = "
    for var_n in name:
        varname += " " + var_n

    with open(tecname, 'w') as f:
        f.write(varname + '\n')
        f.write(f'ZONE N = {mesh.get_num_points()}, E = {mesh.get_num_elements()}\n')
        f.write('DATAPACKING=Point, ZONETYPE=FETETRAHEDRON\n')
        np.savetxt(string, x, fmt=len(name) * ' %e')
        f.write(string.getvalue())
        string.truncate(0)
        string.seek(0)
        np.savetxt(string, connectivity + 1, fmt="%d")
        f.write(string.getvalue())


def main(arg):
    argv = arg[1:]
    infile = []
    for i, s in enumerate(argv):
        if s.split('.')[-1] == 'pvd':
            infile += [s]
        else:
            break
    outfile = [argv[i]]
    istart, iend, istep = int(argv[i + 1]), int(argv[i + 2]), int(argv[i + 3])
    meshfile = infile[0]
    meshfile = meshfile[:-4] + str(istart) + '.pvd'
    print('Preparing mesh', flush=True)
    mesh = PVDMesh(meshfile)
    print('Mesh ready', flush=True)

    # Data container
    container = [PVDData(mesh) for i in range(len(infile))]
    for i in range(istart, iend, istep):
        print('Reading:')
        for j, filename in enumerate(infile):
            ifname = filename[:-4] + str(i) + '.pvd'
            print('  ' + ifname)
            container[j].read_pvd(ifname)
        ofname = outfile[0][:-4] + str(i) + '.tec'
        print('Writing:')
        print('  ' + ofname)
        writeTEC(ofname, mesh, container)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
