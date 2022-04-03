import os
import pickle
import pandas as pd
import datetime as dt
import numpy as np
import netCDF4 as nc


class snowData():

    def __init__(
        self,
        latlon
    ):
        self.diff = dt.timedelta(hours=9)
        self.interval = dt.timedelta(hours=1)
        self.latlon = latlon
    

    def readData(self, fname):
        with open(fname):
            file = nc.Dataset(fname,mode='r')
            for v in file.variables.keys():
                var = file.variables[v]
                print(var)


    def loadingData(self, year, month):
        if month == 12:
            day_count = 31
        else:
            day_count = (dt.date(year, month+1, 1) - dt.date(year, month, 1)).days
        
        month_era5 = []
        for i in range(0, day_count): 
            day = i+1

            for j in range(0, 6):
                start_date = dt.datetime(year, month, day, 9, 0)
                kst = (start_date + j * self.interval) 
                utc = kst - self.diff

                ym = utc.strftime("%Y%m")
                ymdhm = utc.strftime("%Y%m%d_%H%M")
                mlfile = 'data/{}/ERA5_anal_ml_{}.nc'.format(ym, ymdhm)
                sfcfile = 'data/{}/ERA5_anal_sfc_{}.nc'.format(ym, ymdhm)

                if not os.path.isfile(mlfile) or not os.path.isfile(sfcfile):
                    pass
                else:
                    ml = self.mlDataset(mlfile)              # (23, 2194)
                    sfc = self.sfcDataset(sfcfile)           # (23, 24)
                    data = np.concatenate((ml, sfc), axis=1) # (23, 2218)                 
                    month_era5.append(data)
        return np.array(month_era5)
                


    def mlDataset(self, fname): # 22 x / 22 y concatenate = 1
 
        mls = []
        with open(fname):
            ds = nc.Dataset(fname, mode='r')
            for point, i, j in self.latlon:
                time = ds.variables['time'] #0
                time_ = nc.num2date(time[:], time.units, time.calendar) 
                time_ = dt.datetime.strptime(str(time_[0]),'%Y-%m-%d %H:%M:%S') 
                time_ = int((time_+self.diff).strftime("%Y%m%d%H")) # time = 0 / point = 1
                crwc = ds.variables['crwc'][:, :, i, j].reshape(-1) # 2-138
                cswc = ds.variables['cswc'][:, :, i, j].reshape(-1) # 139-275
                etadot = ds.variables['etadot'][:, :, i, j].reshape(-1) # 276-412
                z = ds.variables['z'][:, :, i, j].reshape(-1) # 413-549
                t = ds.variables['t'][:, :, i, j].reshape(-1) # 550-686
                q = ds.variables['q'][:, :, i, j].reshape(-1) # 685-823
                w = ds.variables['w'][:, :, i, j].reshape(-1) # 822-960
                vo = ds.variables['vo'][:, :, i, j].reshape(-1) # 961-1097
                lnsp = ds.variables['lnsp'][:, :, i, j].reshape(-1) # 1098-1234
                d = ds.variables['d'][:, :, i, j].reshape(-1) # 1235-1371
                u = ds.variables['u'][:, :, i, j].reshape(-1) # 1372-1508
                v = ds.variables['v'][:, :, i, j].reshape(-1) # 1509-1645
                o3 = ds.variables['o3'][:, :, i, j].reshape(-1) # 1646-1782
                clwc = ds.variables['clwc'][:, :, i, j].reshape(-1) # 1783-1919
                ciwc = ds.variables['ciwc'][:, :, i, j].reshape(-1) # 1920-2056
                cc = ds.variables['cc'][:, :, i, j].reshape(-1) # 2057-2193
                ml = np.hstack((time_, point, crwc, cswc, etadot, z, t, q, w, vo, 
                                lnsp, d, u, v, o3, clwc, ciwc, cc)) # (1, 2194)
                mls.append(ml)
            
        """
        mldict = {
                'time': time_, 'crwc': crwc, 'cswc': cswc, 
                'etadot': etadot, 'z': z, 't': t, 'q': q, 'w': w, 
                'vo': vo, 'lnsp': lnsp, 'd': d, 'u': u, 'v': v, 
                'o3': o3, 'clwc': clwc, 'ciwc': ciwc, 'cc': cc
                }
        """
        return np.array(mls)


    def sfcDataset(self, fname):
        sfcs = []
        with open(fname):
            ds = nc.Dataset(fname, mode='r')
            for point, i, j in self.latlon:
                lsm = ds.variables['lsm'][:, i, j].reshape(-1) # 2194
                siconc = ds.variables['siconc'][:, i, j].reshape(-1) # 2195
                asn = ds.variables['asn'][:, i, j].reshape(-1) # 2196
                rsn = ds.variables['rsn'][:, i, j].reshape(-1) # 2197
                sst = ds.variables['sst'][:, i, j].reshape(-1) # 2198
                sp = ds.variables['sp'][:, i, j].reshape(-1) # 2199
                sd = ds.variables['sd'][:, i, j].reshape(-1) # 2200
                msl = ds.variables['msl'][:, i, j].reshape(-1) # 2201
                blh = ds.variables['blh'][:, i, j].reshape(-1) # 2202
                tcc = ds.variables['tcc'][:, i, j].reshape(-1) # 2203
                u10 = ds.variables['u10'][:, i, j].reshape(-1) # 2204
                v10 = ds.variables['v10'][:, i, j].reshape(-1) # 2205
                t2m = ds.variables['t2m'][:, i, j].reshape(-1) # 2206
                d2m = ds.variables['d2m'][:, i, j].reshape(-1) # 2207
                lcc = ds.variables['lcc'][:, i, j].reshape(-1) # 2208 
                mcc = ds.variables['mcc'][:, i, j].reshape(-1) # 2209
                hcc = ds.variables['hcc'][:, i, j].reshape(-1) # 2210
                skt = ds.variables['skt'][:, i, j].reshape(-1) # 2211
                swvl1 = ds.variables['swvl1'][:, i, j].reshape(-1) # 2212
                swvl2 = ds.variables['swvl2'][:, i, j].reshape(-1) # 2213
                swvl3 = ds.variables['swvl3'][:, i, j].reshape(-1) # 2214
                swvl4 = ds.variables['swvl4'][:, i, j].reshape(-1) # 2215
                stl1 = ds.variables['stl1'][:, i, j].reshape(-1) # 2216
                stl2 = ds.variables['stl2'][:, i, j].reshape(-1) # 2217
                stl3 = ds.variables['stl3'][:, i, j].reshape(-1) # 2218
                stl4 = ds.variables['stl4'][:, i, j].reshape(-1) # 2219
                sfc = np.hstack((lsm, siconc, asn, sst, sp, sd, msl, blh, tcc, 
                                u10, t2m, d2m, lcc, mcc, hcc, skt, swvl1,
                                swvl2, swvl3, swvl4, stl1, stl2, stl3, stl4)) #(1, 24)
                sfcs.append(sfc)
        """
        sfcdict = {
                'lsm': lsm, 'siconc': siconc, 'asn': asn, 
                'rsn': rsn, 'sst': sst, 'sp': sp, 'sd': sd, 
                'msl': msl, 'blh': blh, 'tcc': tcc, 'u10': u10, 
                'v10': v10, 't2m': t2m, 'd2m': d2m, 'lcc': lcc, 
                'mcc': mcc, 'hcc': hcc, 'skt': skt, 'swvl1': swvl1, 
                'swvl2': swvl2, 'swvl3': swvl3,'swvl4': swvl4, 
                'stl1': stl1, 'stl2': stl2, 'stl3': stl3, 'stl4': stl4
                }
        """
        return np.array(sfcs)

            
if __name__ == "__main__":
    index = pd.read_csv("data_index.csv")
    era_latlon_index = []
    for idx, row in index.iterrows():
        era_latlon_index.append([row[0], row[3], row[4]])
    y, m = 2018, 1
    down_date = dt.datetime(y, m, 1, 0, 0)
    ym = down_date.strftime("%Y%m") + '.npy'
    save_path = os.path.join('data/prepro', ym)
    sd = snowData(era_latlon_index)
    data = sd.loadingData(y, m)
    a = data[0,0]
    np.save(save_path, data)