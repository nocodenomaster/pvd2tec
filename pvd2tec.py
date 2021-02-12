
from io import StringIO
import numpy as np

class PVDFile():
    def __init__(self, filename):
        self._quantity_name = ""
        self._file = []
        self._mesh_dim = -1
        self._quantity_dim = -1
        self.get_file_info(filename)
    def xml_line_decoder(self, line):
        #clean #0
        id0 = line.index('<')
        id1 = line.index('>')
        clean0 = line[id0:id1].split(' ')
        #clean #1
        array_dict = {}
        for l in clean0:
            if(l.find('=') == -1):
                continue
            else:
                pair = l.split('=')
                pair[1] = pair[1].replace("\"", "")
                array_dict[pair[0]] = pair[1]
        return array_dict
    def get_file_info(self, pvd_name):
        split_id = pvd_name.rfind('/')
        pathname, filename = pvd_name[:split_id + 1], pvd_name[split_id + 1:]
        #open pvd find pvtu
        isfind = 0
        with open(pvd_name, 'r') as fpvd:
            content_fpvd = fpvd.readlines()
        for line in content_fpvd:
            if (line.find('file') == -1):
                continue
            else:
                clean_dict = self.xml_line_decoder(line)
                pvtu_name = clean_dict['file']
                isfind = 1
        if isfind == 0:
            raise RuntimeError('can\'t find *.pvtu')
        #open pvtu find vtu
        with open(pathname + pvtu_name, 'r') as fpvtu:
            content_fpvtu = fpvtu.readlines()
            if any('Scalars' in line for line in content_fpvtu):
                quantity_type = 'Scalars'
            elif any('Vectors' in line for line in content_fpvtu):
                quantity_type = 'Vectors'
            else:
                raise RuntimeError('Unknown quantity type, not scalar and vector.')
            #quantity name
            for line in content_fpvtu:
                name_id = line.find(quantity_type)
                src_id = line.find('Source')
                if (self._quantity_name and self._quantity_dim == -1):
                    dim = int(self.xml_line_decoder(line)['NumberOfComponents'])
                    self._quantity_dim = 1 if dim == 0 else dim
                if (name_id >= 0):
                    self._quantity_name = self.xml_line_decoder(line)[quantity_type]
                elif (src_id >= 0):
                    self._file.append(pathname + self.xml_line_decoder(line)['Source'])
    def get_quantity_name(self):
        dim = self._quantity_dim
        if dim > 1:
            name = self._quantity_name
            namelist = [name + str(i) for i in range(dim)]
            namelist.append('|' + name + '|')
            return namelist
        return self._quantity_name
    def get_quantity_dim(self):
        return self._quantity_dim
    def get_file(self):
        return self._file
class PVDMesh(PVDFile):
    def __init__(self, filename):
        super().__init__(filename)
        self._filelist = self.get_file()
        self._point, self._connectivity = self.read_vtu()
    def read_vtu(self):
        point = np.zeros((0, 3))
        connectivity = np.zeros((0, 4))
        num_points, num_elements = 0, 0
        for filename in self._filelist:
            with open(filename, 'r') as fvtu:
                context_fvtu = fvtu.readlines()
                clean_dict0 = self.xml_line_decoder(context_fvtu[3])
                num_local_points = int(clean_dict0['NumberOfPoints'])
                num_local_elements = int(clean_dict0['NumberOfCells'])
                num_points += num_local_points
                num_elements += num_local_elements
                point_str = context_fvtu[5].replace('>', '<').split('<')[2].split(' ')
                newpoint = np.array([float(p) for p in point_str if p != ''])
                point = np.vstack((point, newpoint.reshape(-1, 3)))
                connectivity_str = context_fvtu[8].replace('>', '<').split('<')[2].split(' ')
                newconnectivity = np.array([int(c) for c in connectivity_str if c != ''])
                connectivity  = np.vstack((connectivity, num_points + newconnectivity.reshape(-1, 4)))
        return (point, connectivity)
    def get_points(self):
        return self._point
    def get_elements(self):
        return self._connectivity
    def get_num_points(self):
        return self._point.shape[0]
    def get_num_elements(self):
        return self._connectivity.shape[0]
class PVDData(PVDFile):
    def __init__(self, filename):
        super().__init__(filename)
        self._filelist = self.get_file()
        self._quantity_dim = self.get_quantity_dim()
        self._data = self.read_vtu()
    def read_vtu(self):
        dim = self.get_quantity_dim()
        data = np.zeros((0, dim))
        for filename in self._filelist:
            with open(filename, 'r') as fvtu:
                context_fvtu = fvtu.readlines()
                data_str = context_fvtu[13]
                data_str = data_str.replace('>', '<').split('<')[2].split(' ')
                newdata = np.array([float(q) for q in data_str if q != ''])
                data = np.vstack((data, newdata.reshape(-1, dim)))
        return data
    def get_data(self):
        data = self._data
        # get norm
        if data.shape[1] > 1:
            data_norm = np.linalg.norm(data, 2, axis = 1).reshape((-1, 1))
            return np.hstack((data, data_norm))
        return data
    def get_name(self):
        dim = self.get_quantity_dim()
        name = self.get_quantity_name()
        namelist = []
        if(dim > 1):
            namelist.append("\"" + name + "\"")
            return namelist
        for i in range(self.get_quantity_dim()):
            namelist.append( "\"" + name + str(i) + "\"")
        namelist.append("\"|" + name + "|\"")
        return namelist
def writeTEC(tecname, mesh, datalist):
    string = StringIO()
    x = mesh.get_points()
    name = ['X', 'Y', 'Z']
    connectivity = mesh.get_elements()
    if datalist is PVDData:
        datalist = [datalist]
    for ref in datalist:
        if not isinstance(ref, PVDData):
            raise TypeError(f"ref ({type(ref)}) is not PVDData")
        dim = ref.get_quantity_dim()
        for newname in ref.get_quantity_name():
            name.append(newname) 
        x = np.hstack((x, ref.get_data()))
    varname = "VARIABLES = "
    for var_n in name:
        varname += " " + var_n

    with open(tecname, 'w') as f:
        f.write(varname + '\n')
        f.write(f'ZONE N = {mesh.get_num_points()}, E = {mesh.get_num_elements()}\n')
        f.write('DATAPACKING=Point, ZONETYPE=FETETRAHEDRON\n')
        print(x.shape, name)
        np.savetxt(string, x, fmt = len(name) * ' %e')
        f.write(string.getvalue())
        string.truncate(0)
        string.seek(0)
        np.savetxt(string, connectivity, fmt = "%d")
        f.write(string.getvalue())
if __name__ == "__main__":
   mesh = PVDMesh('../results/tem0.pvd')
   tem = PVDData('../results/tem0.pvd')
   vel = PVDData('../results/vel0.pvd')
   writeTEC('solution.5.tec', mesh, [tem, vel])
   #print(pvd.get_quantity_name())
   #print(pvd.get_quantity_dim())
   #for s in pvd.get_file():
   # print(s)

