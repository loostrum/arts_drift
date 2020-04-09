#!/usr/bin/env python3

import sys
import argparse

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, FK5
from astropy.coordinates.name_resolve import NameResolveError
from astropy.time import Time, TimeDelta

# WSRT location
WSRT_LAT = 52.915184 * u.deg  # = 52:54:54.66
WSRT_LON = 6.60387 * u.deg  # = 06:36:13.93
WSRT_ALT = 16 * u.m
WSRT_LOC = EarthLocation.from_geodetic(WSRT_LON, WSRT_LAT, WSRT_ALT)

# CB size and buffer at start/end of observation
CB_OFFSET = 30 * u.arcmin  # distance between two neighbouring CBs
CB_HPBW = 45 * u.arcmin  # rounded up from 41 arcmin at 1220 MHz
CB_BUFFER = 20 * u.arcmin


def get_scan_duration(ncb, dec):
    """
    Calculate drift scan duration

    :param int ncb: Number of CBs in one scan
    :param astropy.units.quantity.Quantity dec: Source declination
    :return: scan duration (astropy.units.quantity.Quantity)
    """
    # calculate required scan size in arcmin
    scan_size_raw = (ncb - 1) * CB_OFFSET + CB_HPBW + 2 * CB_BUFFER
    # scale by cos(Dec) to get size in RA/HA
    scan_size = scan_size_raw / np.cos(dec)
    # use Earth rotation rate (360 deg per sidereal day) to convert to scan duration
    rate = 360 * u.deg / u.sday
    duration = scan_size / rate
    # return duration in seconds
    return duration.to(u.s)


def ncb_from_row(row):
    """
    Get number of CBs from row index

    :param int row: PAF row
    :return: number of CBs in PAF row (int)
    """
    if row == 0:
        ncb = 1
    elif row in (3, 4, 5):
        ncb = 6
    elif row in (1, 2, 6):
        ncb = 7
    else:
        raise ValueError('Invalid row index: {}'.format(row))
    return ncb


def cbs_from_row(row):
    """
    Get reference CB from row index

    :param int row: PAF row
    :return: reference CB (int)
    """
    row_cbs = {0: (0, 0), 1: (1, 7), 2: (8, 14), 3: (15, 20), 4: (21, 26), 5: (27, 32), 6: (33, 39)}
    try:
        cb_start, cb_end = row_cbs[row]
    except KeyError:
        raise ValueError('Invalid row index: {}'.format(row))
    return cb_start, cb_end


def get_pointing(coord, t):
    """
    Convert J2000 coordinates to apparent HA, Dec
    Add offset so source starts 1/2 CB HPBW + buffer away from CB centre

    :param astropy.coordinate.SkyCoord coord: J2000 source coordinates
    :param astropy.time.Time t: UTC time, including location
    :return: ha, dec (astropy.units.quantity.Quantity)
    """
    # define coordinate system using equinox of date (as opposed to J2000)
    coord_system = FK5(equinox='J{}'.format(t.decimalyear))
    # convert source coordinates to apparent coordinates
    source_apparent = coord.transform_to(coord_system)
    dec = source_apparent.dec
    # calculate HA from RA and LST
    lst = t.sidereal_time('apparent')
    ha = lst - source_apparent.ra
    # pointing should be such that source is at 1/2 HBPW + buffer away from ref CB
    # scale by cos(dec) to get correct size in HA
    offset = (.5 * CB_HPBW + CB_BUFFER) / np.cos(dec)
    # point slightly further west (increase HA) so source is east of beam pattern by this amount
    ha += offset

    # ensure ha is in correct range
    if ha > 180 * u.deg:
        ha -= 360 * u.deg
    elif ha < -180 * u.deg:
        ha += 360 * u.deg
    # check if in range of equatorial mount
    if ha > 90 * u.deg:
        print('WARNING: found HA > 90 deg, source is not visible')
    elif ha < -90 * u.deg:
        print('WARNING: found HA < 90 deg, source is not visible')

    return ha, dec


def get_schedule_line(src, tstart, dur, ha, dec, cb_start, cb_end, cb_ref, freq):
    # end time
    tend = tstart + dur
    # split into date and time
    date1, time1 = tstart.isot.split('T')
    date2, time2 = tend.isot.split('T')
    # remove sub-second part
    time1 = time1.split('.')[0]
    time2 = time2.split('.')[0]

    # construct source name
    cb_start_str = '{:02d}'.format(cb_start)
    # cb end is only included in source name if different from cb start
    if cb_end != cb_start:
        cb_end_str = '{:02d}'.format(cb_end)
    else:
        cb_end_str = ''
    source_name = '{}drift{}{}'.format(src, cb_start_str, cb_end_str)

    # convert HA and Dec to hh:mm:ss / dd:mm:ss strings
    ha_str = ha.to_string(unit=u.hourangle, sep=':', pad=True, precision=4)
    dec_str = dec.to_string(unit=u.deg, sep=':', pad=True, precision=4)

    # schedule line
    # source,ra,ha,dec,date1,time1,date2,time2,centfreq,weight,sbeam,ebeam,pulsar,beam'
    schedule_line = '{},,{},{},{},{},{},{},{},square_39p1,0,39,False,{:02d}\n'.format(source_name,
                                                                                      ha_str, dec_str,
                                                                                      date1, time1,
                                                                                      date2, time2,
                                                                                      freq, cb_ref)

    return schedule_line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARTS drift scan scheduler',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog='Example usage:\n'
                                            'Full PAF drift scans:\n'
                                            './artsdrift.py --csv output.csv --src 3C48 --tstart 2020-01-01T00:00:00'
                                            '\n\n'
                                            'Only CBs 00 and 33:\n'
                                            './artsdrift.py --csv output.csv --src 3C48 --tstart 2020-01-01T00:00:00'
                                            ' --beams 0 33\n\n'
                                            'Only CBs 01 to 07 and 27 to 32:\n'
                                            './artsdrift.py --csv output.csv --src 3C48 --tstart 2020-01-01T00:00:00'
                                            ' --rows 1 5\n\n'
                                            'Note: --rows and --beams are mutually exclusive')
    parser.add_argument('--output', type=argparse.FileType('a'), default=sys.stdout,
                        help='Output file to append schedule to (optional, else print to stdout)')
    parser.add_argument('--source', required=True, help='source name, can be any name resolvable by CDS')
    parser.add_argument('--tstart', required=True, help='UTC start time in ISOT format, e.g. 2020-01-01T00:00:00')
    parser.add_argument('--wait_time', type=float, default=3., help='Wait time between observations in minutes'
                                                                    ' (Default: %(default)s)')
    parser.add_argument('--freq', type=int, default=1370, help='Central frequency in MHz (Default: %(default)s)')
    parser.add_argument('--noheader', action='store_true', help='Disable schedule header')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rows', type=int, nargs='+', help='PAF row(s) to drift through (default: all). Row indices:\n'
                                                           '0: CB 00\n'
                                                           '1: CB 01 to 07\n'
                                                           '2: CB 08 to 14\n'
                                                           '3: CB 15 to 20\n'
                                                           '4: CB 21 to 26\n'
                                                           '5: CB 27 to 32\n'
                                                           '6: CB 33 to 39')
    group.add_argument('--beams', type=int, nargs='+', help='CB(s) to drift through, alternative to --rows')

    args = parser.parse_args()

    # parse input rows / beams
    if args.beams:
        rows = False
        inputs = args.beams
    elif args.rows:
        rows = True
        inputs = args.rows
    else:
        # default: all 7 rows
        rows = True
        inputs = range(7)

    # verify inputs are within allowed range
    if rows:
        # 7 rows in total
        label = 'row'
        minval = 0
        maxval = 6
    else:
        # 40 CBs in total
        label = 'beam'
        minval = 0
        maxval = 39

    if np.any(np.array(inputs) > maxval):
        print("ERROR: {} indices cannot be larger than {}".format(label, maxval))
        sys.exit(1)
    elif np.any(np.array(inputs) < minval):
        print("ERROR: {} indices cannot be smaller than {}".format(label, minval))
        sys.exit(1)

    # parse source name
    try:
        src = SkyCoord.from_name(args.source)
    except NameResolveError as e:
        print("ERROR: no coordinates found for {}: {}".format(args.source, e))
        sys.exit(1)

    # check if source is observable
    DECLIM = -35 * u.deg
    if src.dec < DECLIM:
        print("ERROR: source Dec ({}) is not observable by WSRT (limit: {})".format(src.dec, DECLIM))
        sys.exit(1)

    # parse start time
    try:
        tstart = Time(args.tstart, format='isot', scale='utc', location=WSRT_LOC)
    except ValueError as e:
        print("ERROR: could not parse start time {}: {}".format(args.tstart, e))
        sys.exit(1)

    # init schedule
    if args.noheader:
        schedule = ''
    else:
        schedule = 'source,ra,ha,dec,date1,time1,date2,time2,centfreq,weight,sbeam,ebeam,pulsar,beam\n'

    # loop over beams / rows and add line to schedule for each observation
    for value in inputs:
        # get number of CBs: determined from row index, or one if doing single CBs
        # start and end CB are first and last of row, reference beam is last bea
        if rows:
            ncb = ncb_from_row(value)
            cb_start, cb_end = cbs_from_row(value)
            cb_ref = cb_end
        else:
            ncb = 1
            cb_start = value
            cb_end = value
            cb_ref = value

        # get scan duration
        duration = get_scan_duration(ncb, src.dec)

        # get pointing
        pointing_ha, pointing_dec = get_pointing(src, tstart)

        # add line to schedule
        schedule += get_schedule_line(args.source, tstart, duration, pointing_ha, pointing_dec, cb_start, cb_end,
                                      cb_ref, args.freq)

        # start time of next observation
        tstart += duration + args.wait_time * u.minute

    # print / save schedule
    args.output.write(schedule)
    args.output.close()
