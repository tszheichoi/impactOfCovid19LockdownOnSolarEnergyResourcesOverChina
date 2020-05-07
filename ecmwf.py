#!/usr/bin/env python
#-*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import netCDF4
# import pathsSpecifications
# import reveal
import mathsUtilities
import collections

class batchECMWFMeasurements:
	def __init__(self, allPathsDict, verbose = False, maxNumberToStoreInTemp = 2):
		self.verbose, self.allPathsDict, self.maxNumberToStoreInTemp = verbose, allPathsDict, maxNumberToStoreInTemp
		self.cache = collections.OrderedDict()

	def getSurfaceObject(self, time):
		if isinstance(time, datetime):
			year = str(time.year)
			month = '{:02d}'.format(time.month)
			yearMonthString = year + month
		else:
			yearMonthString = time
		if yearMonthString in self.cache:
			return self.cache[yearMonthString]
		else:
			if yearMonthString in self.allPathsDict:
				print('returning key %s: %s'%(yearMonthString, self.allPathsDict[yearMonthString]))
				currentObject = surfaceLevel(str(self.allPathsDict[yearMonthString]))
				self.cache[yearMonthString] = currentObject
				if len(self.cache.keys()) > self.maxNumberToStoreInTemp:
					self.cache.popitem(False)
				return currentObject
			else:
				print('requested date (%s) is not in valid range.'%(yearMonthString))
				return None
				
class surfaceLevel:
	def __init__(self, path, verbose = False):
		self.verbose = verbose
		if self.verbose: print('reading surface level data...')
		f = netCDF4.Dataset(path, scale = True) # reads in CAMS data, apply scaling and offset as needed
		# one dimensinoal data
		self.time = np.asarray([float(t) for t in f.variables['time'][:]]) #time in hours since 1900-01-01 00:00:0.0
		epoch = datetime(1900, 1, 1, 00, 00, 00) #convert time to a datetime object
		self.time = epoch + np.vectorize(timedelta)(hours = self.time) #convert to datetime first
		# multi dimensional data
		self.lat = np.asarray(f.variables['latitude'][:]) #list of all latitudes, keeping as numpy arrays!
		self.lon = np.asarray(f.variables['longitude'][:])
		self.aod469 = np.asarray(f.variables['aod469'][:])
		self.aod550 = np.asarray(f.variables['aod550'][:])
		self.aod670 = np.asarray(f.variables['aod670'][:])
		self.aod865 = np.asarray(f.variables['aod865'][:])
		self.aod1240 = np.asarray(f.variables['aod1240'][:])
		self.aods = {469: self.aod469, 550: self.aod550, 670: self.aod670, 865: self.aod865, 1240: self.aod1240}
		self.aodWvls = [469, 550, 670, 865, 1240]
		
		try:
			self.no2 = np.asarray(f.variables['tcno2'][:])
		except: 
			self.no2 = None
			print('no total columnar no2 found in file.')

		try:
			self.o3 = np.asarray(f.variables['gtco3'][:])
		except: 
			self.o3 = None
			print('no total columnar ozone found in file.')

		self.species = {'no2': self.no2, 'aerosol': self.aod550, 'o3': self.o3}

		if self.verbose: print('> complete.')

	def getTimeList(self):
		return self.time

	def getLatList(self):
		return self.lat

	def getLonList(self):
		return self.lon

	def getNitrogenDioxide(self, time):
		if not isinstance(self.no2, np.ndarray):
			raise Exception('cannot find no2 in file!')
		temporalIndex = self.getTemporalIndex(time)
		return self.no2[temporalIndex, :, :]

	def getTemporalIndex(self, time):
		return np.argmin(np.asarray(np.abs(self.getTimeList() - time)))

	def getClosestLonLat(self, lon, lat):
		[lonIndex, latIndex] = self.getSpatialIndex(lon, lat)
		return [self.getLonList()[lonIndex], self.getLatList()[latIndex]]

	def getSpatialIndex(self, lon, lat):
		lonIndex = (np.abs(self.getLonList() - lon)).argmin()
		latIndex = (np.abs(self.getLatList() - lat)).argmin()
		return [lonIndex, latIndex]

	def getAODTimeSeries(self, wavelength, lon, lat):
		[lonIndex, latIndex] = self.getSpatialIndex(lon, lat)
		if int(wavelength) in self.aods.keys(): # these are directly available
			return self.aods[int(wavelength)][:, latIndex, lonIndex]
		else:
			raise Exception('Cannot fetch timeseries directly at channels that are not directly provided by the IFS')

	def getAODAt(self, wavelength, timeStamp, lon, lat):
		[lonIndex, latIndex] = self.getSpatialIndex(lon, lat)
		temporalIndex = self.getTemporalIndex(timeStamp)
		if int(wavelength) in self.aods.keys(): # these are directly available
			return self.aods[int(wavelength)][temporalIndex, latIndex, lonIndex]
		else: # these need to be interpolated; will do a linear interpolation in log-log space
			aods = [self.aods[w][temporalIndex, latIndex, lonIndex] for w in self.aodWvls]
			if all([a > 0.0 for a in aods]):
				return mathsUtilities.log_interp1d(self.aodWvls, aods)(wavelength)
			elif any([a < 0.00001 for a in aods]):
				return 0.0
			else:
				return mathsUtilities.log_interp1d([w for w, a in zip(self.aodWvls, aods) if a != 0.0], [a for a in aods if a != 0.0])(wavelength)

	def getAngstromExponent(self, wavelengthShort, wavelengthLong, timeStamp, lon, lat): #compute the angstrom parameter between the two given wavelengths
		assert self.getAODAt(wavelengthLong, timeStamp,  lon, lat) != 0.0
		return np.log((self.getAODAt(wavelengthShort, timeStamp, lon, lat)/self.getAODAt(wavelengthLong, timeStamp, lon, lat)))/np.log(float(wavelengthLong)/float(wavelengthShort))

	def getAverage(self, species, fromDate, toDate, steps = ['00', '03', '06', '12', '15', '18']):
		print('computing avg for %s from %s to %s'%(species, str(fromDate), str(toDate)))
		temporalMask = (self.time >= fromDate) & (self.time <= toDate) & (np.vectorize(lambda t: True if t.strftime("%H") in steps else False)(self.time))
		filteredSpeices = self.species[species].copy()
		filteredSpeices[filteredSpeices < 0] = np.nan
		return np.nanmean(filteredSpeices[temporalMask, :, :], axis = 0)

	def getArrays(self, species, fromDate, toDate, steps = ['00', '03', '06', '12', '15', '18']):
		print('computing slices %s from %s to %s'%(species, str(fromDate), str(toDate)))
		temporalMask = (self.time >= fromDate) & (self.time <= toDate) & (np.vectorize(lambda t: True if t.strftime("%H") in steps else False)(self.time))
		filteredSpeices = self.species[species].copy()
		filteredSpeices[filteredSpeices < 0] = np.nan
		return filteredSpeices[temporalMask, :, :]

	def getNO2TimeSeries(self, lon, lat, steps = ['00', '03', '06', '12', '15', '18']):
		[lonIndex, latIndex] = self.getSpatialIndex(lon, lat)
		temporalMask = np.vectorize(lambda t: True if t.strftime("%H") in steps else False)(self.time)
		return self.time[temporalMask], self.no2[temporalMask, latIndex, lonIndex]

	def getAOD550TimeSeries(self, lon, lat, steps = ['00', '03', '06', '12', '15', '18']):
		[lonIndex, latIndex] = self.getSpatialIndex(lon, lat)
		temporalMask = np.vectorize(lambda t: True if t.strftime("%H") in steps else False)(self.time)
		invalidValuesMask = self.aod550[:, latIndex, lonIndex] > 0
		mask = invalidValuesMask & temporalMask
		return self.time[mask], self.aod550[mask, latIndex, lonIndex]

if __name__ == "__main__": # these are some test cases to make sure that the aeronet ground based measurement class is working properly
	lon = 143.907168
	lat = -36.735844
	batchObj = batchECMWFMeasurements(pathsSpecifications.camsSurfacePathsOverAustralia)
	targetTime = datetime.utcfromtimestamp(1565532000)
	surfObj = batchObj.getSurfaceObject(targetTime)
	print('the target time is: ', targetTime)
	print('the closest CAMS time slot is: ', surfObj.getTimeList()[surfObj.getTemporalIndex(targetTime)])
	wvls = np.arange(250, 1600, 10)
	reveal.simpleScatter(wvls, [surfObj.getAODAt(w, targetTime, lon, lat) for w in wvls], str(targetTime), r'Wavelengths $nm$', 'Aerosol Optical Depth', 'aerosolOpticalDepthTest', logx = True, logy = True)