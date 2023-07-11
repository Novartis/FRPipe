import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nmrglue as ng
import urllib
import pandas
import re
import glob
import os
import numpy
import csv
import traceback
from scipy.interpolate import UnivariateSpline
import argparse
import sys
import warnings
import requests
import natsort
from PIL import Image
from io import BytesIO
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 4.5})
plt.rcParams.update({'lines.linewidth': 0.4})
plt.rcParams.update({'lines.markersize': 2})
plt.rcParams.update({'lines.markeredgewidth': 0.3})
plt.rcParams.update({'axes.linewidth': 0.3})
plt.rcParams.update({'ytick.major.width': 0.3})
plt.rcParams.update({'ytick.major.size': 2.0})
plt.rcParams.update({'ytick.major.pad': 1.5})
plt.rcParams.update({'ytick.minor.width': 0.1})
plt.rcParams.update({'ytick.minor.size': 1.5})
plt.rcParams.update({'ytick.minor.pad': 1.4})
plt.rcParams.update({'xtick.major.width': 0.3})
plt.rcParams.update({'xtick.major.size': 2.0})
plt.rcParams.update({'xtick.major.pad': 1.5})
plt.rcParams.update({'xtick.minor.width': 0.1})
plt.rcParams.update({'xtick.minor.size': 1.5})
plt.rcParams.update({'ytick.minor.pad': 1.4})
plt.rcParams.update({'legend.frameon': False})


class parameters:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputPath", type=str,
                        help="provide source path that contains NMRData")

    parser.add_argument("-t", "--targetPath", type=str,
                        help="provide target path here. Tables and plots will be written here")

    parser.add_argument("-k", "--KD", type=str,
                        help="provide KD of reporter")

    parser.add_argument("-o", "--overwrite", default=False, action="store_true",
                        help="overwrite targetPath")

    parser.add_argument("-ci", "--confidence", type=float, default=0.2,
                        help="give confidence borders as fraction of NMR-Maxeffect")

    parser.add_argument("-r", "--reporter_ppm", type=float,
                        help="give reporter ppm")

    parser.add_argument("-c", "--control_ppm", type=float,
                        help="give control ppm")

    parser.add_argument("-f", "--FID", default=False, action="store_true",
                        help="read raw FID files")

    parser.add_argument("-v", "--peak_values", default=False, action="store_true",
                        help="use peak values (height) instead of peak volume")

    parser.add_argument("-s", "--num_signals", type=int, default=2,
                        help="number of expected signals in reference")

    parser.add_argument("-rw", "--reporter_search_window_size", type=str, default=None,
                        help="search window in ppm for reporter signals in competition experiments. Default is 10*median reference peak width.")

    parser.add_argument("-cw", "--control_search_window_size", type=str, default=None,
                        help="search window in ppm for control signals in competition experiments. Default is 10*median control peak width.")

    parser.add_argument("-pw", "--proton_window_borders", type=str, default=None,
                        help='borders of proton window (e.g. "-1, 10")')

    parser.add_argument("-psw", "--proton_scaling_window_borders", type=str, default=None,
                        help='borders of proton scaling window (e.g. "5, 11")')

    parser.add_argument("-ns", "--no_scaling", default=False, action="store_true",
                        help="turn of ctrl scaling (not recommended)")

    args = parser.parse_args()

    if args.inputPath:
        srcdir = os.path.abspath(args.inputPath)
    else:
        sys.exit("Error: No inputPath given.")

    if args.targetPath:
        tgtdir = os.path.abspath(args.targetPath)
    else:
        sys.exit("Error: No targetPath given.")

    if os.path.exists(tgtdir) and not args.overwrite:
        sys.exit("Warning: TargetPath exists. use -o switch to overwrite.")
    else:
        if not os.path.exists(tgtdir):
            os.makedirs(tgtdir)

    if args.KD:
        try:
            KD = float(args.KD)
        except:
            sys.exit("KD value is not a number.")
    else:
        sys.exit("Please enter KD value for reporter.")

    cint = args.confidence
    if cint > 0.5:
        sys.exit(
            "fraction for confidence interval is bigger than 0.5. This does not make sense. Please choose a smaller value.")

    if args.reporter_ppm:
        rep_ppm = args.reporter_ppm
    else:
        rep_ppm = None

    if args.control_ppm:
        ctrl_ppm = args.control_ppm
    else:
        ctrl_ppm = None

    if args.reporter_search_window_size:
        try:
            reporter_search_window_size = float(args.reporter_search_window_size)
        except:
            sys.exit("reporter search window is not a number.")
    else:
        reporter_search_window_size = None

    if args.control_search_window_size:
        try:
            control_search_window_size = float(args.control_search_window_size)
        except:
            sys.exit("control search window is not a number.")
    else:
        control_search_window_size = None

    if args.proton_window_borders:
        try:
            proton_window = tuple([int(el) for el in args.proton_window_borders.split(",")])
        except:
            sys.exit("proton window borders input is not valid.")
    else:
        proton_window = None

    if args.proton_scaling_window_borders:
        try:
            proton_scaling_window = tuple([int(el) for el in args.proton_scaling_window_borders.split(",")])
        except:
            sys.exit("proton scaling window borders input is not valid.")
    else:
        proton_scaling_window = None

    raw = args.FID

    if args.peak_values:
        property_name = "value"
    else:
        property_name = "vol"

    nref_sig = args.num_signals

    if args.no_scaling:
        ctrl_scaling = False
    else:
        ctrl_scaling = True


def getSmiles(sample):
    smiles = None
    # dummy function
    # insert code to retreive smiles here
    return smiles


def getDepiction(smiles):
    # dummy function
    # add coded to render depiction from smiles here

    return None


def read_bruker_rawdata(dirname):
    raw_dirname = os.path.dirname(os.path.dirname(dirname))
    dic, data = ng.bruker.read(raw_dirname)
    if dic["acqus"]["NUC1"] == "19F":

        data = ng.bruker.remove_digital_filter(dic, data)

        data = data[0:8 * 1024]
        data = ng.proc_base.sine(data, off=0.42, end=1, pow=6)
        data = ng.proc_base.zf_size(data, 65536)

        data = ng.proc_base.fft(data)
        data = ng.proc_autophase.autops(data, "acme")

        data = ng.proc_base.di(data)
        data = ng.proc_base.rev(data)

        data = ng.process.proc_bl.baseline_corrector(data, wd=20)
        data = ng.process.proc_base.smo(data, 2)
    else:
        dic, data = ng.bruker.read_pdata(dirname)
    return (dic, data)


def convert_bruker_nmrdata(dirname, raw=False, nref_sig=2):
    def parse_title_file(title_file_name):
        titleDict = dict()
        try:
            titlefile = open(title_file_name, "r")

        except:
            print("Cannot open %s" % title_file_name)
            return titleDict

        for line in titlefile:
            if re.match("^Ligand:", line):
                titleDict["lig_conc"] = float(re.search("(?<=Ligand: )[0-9.]{1,5}(?= uM)", line).group())
                titleDict["lig_name"] = re.search("(?<= uM )[A-Za-z0-9\-]*", line).group()

            if re.match("^Reporter:", line):
                titleDict["rep_conc"] = float(re.search("(?<=Reporter: )[0-9.]{1,5}(?= uM)", line).group())
                titleDict["rep_name"] = re.search("(?<= uM )[A-Z0-9\-]*", line).group()

            if re.match("^Protein:", line):
                titleDict["prot_conc"] = float(re.search("(?<=Protein: )[0-9.]{1,5}(?= uM)", line).group())
                titleDict["prot_name"] = re.search("(?<= uM )[A-Za-z0-9\-()\_]*", line).group()
                if re.search("(?<=PN)[0-9]*", line):
                    titleDict["prot_batchID"] = "PN%s" % re.search("(?<=PN)[0-9]*", line).group()
                elif re.search("(?<=APP-)[0-9]*", line):
                    titleDict["prot_batchID"] = "APP-%s" % re.search("(?<=APP-)[0-9]*", line).group()

        return titleDict

    try:
        # load bruker data
        if raw:
            dic, data = read_bruker_rawdata(dirname)
        else:
            dic, data = ng.bruker.read_pdata(dirname)

        # generate universal dictionary and conversion objects
        udic = ng.bruker.guess_udic(dic, data)
        uc0 = ng.fileiobase.uc_from_udic(udic, dim=0)

    except:
        return (False, False, False)

    # generate ppm values for axis indices
    ax0 = [uc0.ppm(i) for i in range(len(data))]

    # copy array to dataframes
    dt = pandas.DataFrame(data=data)
    # replace indices by ppm values
    dt.set_axis(axis=0, labels=ax0, inplace=True)

    line = dict()

    title_file_name = os.path.join(dirname, "title")
    titleDict = dict()

    try:
        titleDict = parse_title_file(title_file_name)
        fn = dict()
        fn["a"] = dirname.split("/")[-4]
        fn["b"] = dirname.split("/")[-3]
        line["name"] = "%s_%s" % (fn["a"], fn["b"])
        line["expnumber"] = fn["b"]
        for key in (
                "lig_conc", "lig_name", "rep_conc", "rep_name", "prot_name", "prot_conc", "prot_batchID"):
            if key in titleDict.keys():
                line[key] = titleDict[key]
            else:
                line[key] = "NA"

    except:
        # build filenames
        print("Cannot parse title dict for %s" % title_file_name)

    if titleDict:
        is_ref = False
        if line["lig_conc"] == "NA" or line["lig_conc"] == 0:
            is_ref = True
    else:
        is_ref = False

    if is_ref:
        if raw:
            n = 5
        else:
            n = 0.5
        threshold = float(numpy.std(data) * n)
        peaks = ng.peakpick.pick(data, threshold)
        while len(peaks) > nref_sig:
            n = n + 0.25
            threshold = float(numpy.std(data) * n)
            peaks = ng.peakpick.pick(data, threshold)
    else:
        # collect peak data
        n = 0.25
        threshold = float(numpy.std(data) * n)
        peaks = ng.peakpick.pick(data, threshold)
        while len(peaks) > 5:
            n = n + 0.25
            threshold = float(numpy.std(data) * n)
            peaks = ng.peakpick.pick(data, threshold)

    line["peak_ppm"] = list()
    line["peak_vol"] = list()
    line["peak_width"] = list()
    line["peak_value"] = list()

    for peak in peaks:
        line["peak_ppm"].append(uc0.ppm(peak[0]))
        line["peak_vol"].append(peak[-1])
        line["peak_width"].append(numpy.abs(uc0.ppm(peak[0] - peak[2] / 2) - uc0.ppm(peak[0] + peak[2] / 2)))
        line["peak_value"].append(data[int(peak[0])])

    line["D2"] = dic["acqus"]["D"][2]
    line["NS"] = dic["acqus"]["NS"]
    line["EXP"] = dic["acqus"]["EXP"]
    line["TE"] = dic["acqus"]["TE"]
    line["PULPROG"] = dic["acqus"]["PULPROG"]
    line["DATE"] = dic["acqus"]["DATE"]
    line["NUC1"] = dic["acqus"]["NUC1"]

    return (True, line, dt)


def getDataDirs(srcdir):
    dataDirs = glob.glob(os.path.join(srcdir + "/*" + "/pdata/1"))
    return (dataDirs)


def NMR_function(r_values, Io, KIvalues):
    # Function to calculate fraction of reporter molecule bound to receptor
    # https://pubs.acs.org/doi/pdf/10.1021/ja0542385

    KD = numpy.float64(r_values["KD"])
    Eo = numpy.float64(r_values["Eo"])
    Lo = numpy.float64(r_values["Lo"])
    if isinstance(KIvalues, (int, float)):
        KIvalues = [KIvalues, ]

    rI = list()
    for KI in KIvalues:
        C1 = numpy.float64(KD + KI + Lo + Io - Eo)  # a
        C2 = numpy.float64(Io - Eo)
        C3 = numpy.float64(Lo - Eo)
        C4 = numpy.float64(C2 * KD + C3 * KI + KD * KI)  # b

        TA = numpy.float64(2 * numpy.sqrt(numpy.power(C1, 2) - 3 * C4) *
                           numpy.cos(numpy.arccos((-2 * numpy.power(C1, 3)
                                                   + 9 * C1 * C4 - 27 * (-KD * KI * Eo))
                                                  / numpy.float64(
                               2 * numpy.sqrt(numpy.power((numpy.power(C1, 2) - 3 * C4), 3))))
                                     / numpy.float64(3)) - C1)

        rI.append(TA / numpy.float64(3 * KD + TA))
    if len(rI) == 1:
        return numpy.float64(rI[0])
    else:
        return rI


def rI_conv(rIvalues, norm_factor):
    # function to convert reporter intensity to expected signal

    if isinstance(rIvalues, (int, float)):
        rIvalues = [rIvalues, ]
    Xint = list()
    for rI in rIvalues:
        Xint.append(1 - rI / norm_factor)
    if len(Xint) == 1:
        return float(Xint[0])
    else:
        return Xint


def iter_fit(NMR_signal, r_values, Io, norm_factor):
    # function to calculate fit

    x1 = -10
    x2 = 7
    step = 0.01
    tol = pow(10, -20)

    x_val = numpy.arange(x1, x2, step)
    y_val = rI_conv(NMR_function(r_values, Io, numpy.power(10, x_val)), norm_factor)
    index = numpy.argmax(numpy.array(y_val) <= NMR_signal)
    if y_val[-1] < NMR_signal < y_val[0]:
        while NMR_signal - y_val[index] > tol:
            x1 = x_val[index - 1]
            x2 = x_val[index + 1]
            step = step / 10
            x_val = numpy.arange(x1, x2, step)
            y_val = rI_conv(NMR_function(r_values, Io, numpy.power(10, x_val)), norm_factor)
            index = numpy.argmax(numpy.array(y_val) <= NMR_signal)

        return numpy.power(10, x_val[index])
    else:
        return None


# plotting function
def plot_curves(ax, ligand, reporter, protein,
                r_values, Ki_list, norm_factor, colmap,
                lower_limit=0.4, upper_limit=0.8):
    def extract_Io(el):
        return el["Io [uM]"]

    ax.plot()
    ax.set(xlabel="Ki [uM]")
    ax.set(ylabel="expected rel. reporter signal [AU]")
    ax.set(xscale="log")

    # plot different Io values
    legend_text = list()
    Ki_list_red = [el for el in Ki_list if "Io [uM]" in el.keys()]
    Ki_list_red.sort(key=extract_Io)
    max_Io = max(set([float(el["Io [uM]"]) for el in Ki_list_red]))
    mean_Io = numpy.mean(list(set([el["Io [uM]"] for el in Ki_list_red])))
    mean_NMR_signal = lower_limit + ((upper_limit - lower_limit) / 2)
    mean_Ki = iter_fit(mean_NMR_signal, r_values, mean_Io, norm_factor)
    if not mean_Ki:
        mean_Ki = 100

    x1 = round(numpy.log10(mean_Ki), 0) - 4
    x2 = round(numpy.log10(mean_Ki), 0) + 4

    x_val = numpy.arange(x1, x2 + 1, 0.001)
    x = (pow(10, x1), pow(10, x2))
    y = (0, 1)

    ax.set(ylim=y)
    ax.set(xlim=x)

    for Io_value in set([el["Io [uM]"] for el in Ki_list_red]):
        ax.plot(pow(10, x_val), rI_conv(NMR_function(r_values, Io_value, pow(10, x_val)), norm_factor),
                color=plt.cm.pink(float(Io_value) / (max_Io * 2)))

    for Ki_row in Ki_list_red:
        colmap_key = [el for el in colmap.keys()
                      if re.match("^[\ ]?%3.i_%i" % (Ki_row["Io [uM]"], int(Ki_row["expnumber"])), el)][0]
        color = colmap[colmap_key]
        if Ki_row["exclude"]:
            marker_style = "x"
            markeredgecolor = color
            if 0 < Ki_row["Ki [uM]"] < 100000:
                label = "%4.i uM, Ki %.1f uM, #%s" % (Ki_row["Io [uM]"], Ki_row["Ki [uM]"], Ki_row["expnumber"])
                ax.text(1.1 * Ki_row["Ki [uM]"], Ki_row["NMR_signal [AU]"] + 0.002, "%.1f" % Ki_row["Ki [uM]"],
                        ha="left",
                        va="bottom", fontsize="small")
            else:
                label = "%4.i uM, Ki %.1f uM, #%s" % (Ki_row["Io [uM]"], numpy.nan, Ki_row["expnumber"])
        else:
            marker_style = "o"
            markeredgecolor = "black"
            label = "%4.i uM, Ki %.1f uM, #%s" % (Ki_row["Io [uM]"], Ki_row["Ki [uM]"], Ki_row["expnumber"])
            ax.text(1.1 * Ki_row["Ki [uM]"], Ki_row["NMR_signal [AU]"] + 0.002, "%.1f" % Ki_row["Ki [uM]"], ha="left",
                    va="bottom", fontsize="small")

        ax.plot(Ki_row["Ki [uM]"], Ki_row["NMR_signal [AU]"],
                marker=marker_style, markerfacecolor=color, markeredgecolor=markeredgecolor,
                color=plt.cm.pink(float(Ki_row["Io [uM]"]) / (max_Io * 2)),
                label=label)

    ax.legend(fontsize="small", numpoints=1)

    ax.text(pow(10, x2 - 2.7), 0.05,
            "mean Ki: %s uM SD: %s" % (Ki_list[-1]["mean Ki [uM]"], Ki_list[-1]["SD Ki [uM]"]),
            fontsize="small")

    # add border lines for detection limits
    ax.plot(pow(10, x_val), upper_limit + 0 * x_val, color="red", ls="dotted")
    ax.plot(pow(10, x_val), lower_limit + 0 * x_val, color="red", ls="dotted")

    ax.plot(Ki_list[-1]["mean Ki [uM]"] + 0 * numpy.arange(0, 1.1, 0.1), numpy.arange(0, 1.1, 0.1), color="black",
            ls="dotted")

    if type((Ki_list[-1]["SD Ki [uM]"])) == float or type((Ki_list[-1]["SD Ki [uM]"])) == int:
        ax.plot(Ki_list[-1]["mean Ki [uM]"] + Ki_list[-1]["SD Ki [uM]"] + 0 * numpy.arange(0, 1.1, 0.1),
                numpy.arange(0, 1.1, 0.1), color="blue",
                ls="dotted")
        ax.plot(Ki_list[-1]["mean Ki [uM]"] - Ki_list[-1]["SD Ki [uM]"] + 0 * numpy.arange(0, 1.1, 0.1),
                numpy.arange(0, 1.1, 0.1), color="blue",
                ls="dotted")


def calc_Ki_value(r_values={"KD": 150, "Eo": 3, "Lo": 150},
                  exp_data=[{'Io_value': 150.0, 'NMR_signal': 0.4938957, 'rel_rep_signal': 0.9, 'expnumber': '551'}],
                  NMR_max_effect=0.2,
                  lower_limit=0.4, upper_limit=0.8,
                  input_table=None):
    # calculate some values for conversion
    rI_max = NMR_function(r_values, max([el["Io_value"] for el in exp_data]), pow(10, 6))
    NMR_signal_range = 1 - NMR_max_effect
    norm_factor = rI_max / NMR_signal_range

    if NMR_max_effect > 1:
        print("WARNING: NMR_max_effect is greater than 1. Please check if reporter and control ppms are correct.")

    # calculate fit and Ki
    Ki_list = list()
    for i in range(0, len(exp_data)):
        Ki_row = dict()

        try:
            rfit = iter_fit(exp_data[i]["NMR_signal"],
                            r_values,
                            exp_data[i]["Io_value"],
                            norm_factor)
        except:
            rfit = None
            print("Fitting curve did not work for %s" % exp_data[i]["expnumber"])

        Ki_row["KD"] = r_values["KD"]
        Ki_row["Eo [uM]"] = r_values["Eo"]
        Ki_row["Lo [uM]"] = r_values["Lo"]
        Ki_row["Io [uM]"] = exp_data[i]["Io_value"]
        Ki_row["NMR_signal [AU]"] = exp_data[i]["NMR_signal"]
        Ki_row["rel_rep_signal"] = exp_data[i]["rel_rep_signal"]
        if rfit:
            Ki_row["Ki [uM]"] = float("%.2f" % rfit)
        else:
            Ki_row["Ki [uM]"] = numpy.nan
            print("Fitting curve did not work for %s" % exp_data[i]["expnumber"])

        Ki_row["confidence"] = lower_limit < exp_data[i]["NMR_signal"] < upper_limit
        Ki_row["expnumber"] = exp_data[i]["expnumber"]

        if input_table:
            input_row = [el for el in input_table if el["expnumber"] == exp_data[i]["expnumber"]][0]
            Ki_row["exclude"] = input_row["exclude"]
            Ki_row["comment"] = input_row["comment"]
        else:
            Ki_row["exclude"] = not (lower_limit < exp_data[i]["NMR_signal"] < upper_limit)
            Ki_row["comment"] = ""
        Ki_list.append(Ki_row)

    NMR_signals_in_range = [el for el in Ki_list if not el["exclude"]]
    if [el["Ki [uM]"] for el in NMR_signals_in_range]:
        mean_Ki = float("%.2f" % numpy.mean([el["Ki [uM]"] for el in NMR_signals_in_range]))
        sd_Ki = float("%.2f" % numpy.std([el["Ki [uM]"] for el in NMR_signals_in_range]))
        num_dp = len(NMR_signals_in_range)
        if mean_Ki > 10:
            mean_Ki = round(mean_Ki, 1)
            sd_Ki = round(sd_Ki, 1)
        if mean_Ki > 100:
            mean_Ki = int(round(mean_Ki, 0))
            sd_Ki = int(round(sd_Ki, 0))
        if num_dp < 2:
            sd_Ki = "NA"
        Ki_list.append({"mean Ki [uM]": mean_Ki})
        Ki_list[-1].update({"SD Ki [uM]": sd_Ki})
        Ki_list[-1].update({"# datapoints": num_dp})
        Ki_list[-1].update({"pharon_comment": "SD = %s uM / n = %i" % (sd_Ki, num_dp)})
    else:
        Ki_list.append({"mean Ki [uM]": numpy.NaN})
        Ki_list[-1].update({"SD Ki [uM]": numpy.NaN})
        Ki_list[-1].update({"# datapoints": numpy.NaN})

    return Ki_list, norm_factor


def plot_peaks(ax, data_plot, cols, window, colmap, signal_type, ymax=None):
    ax.plot()
    ax.set(xlabel="ppm")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set(ylabel="%s signal [AU]" % signal_type)
    ax.set(xlim=window)

    if ymax:
        max_value = ymax
    else:
        max_value = numpy.nanmax(data_plot[(data_plot.index < window[0]) & (data_plot.index > window[1])])

    min_value = numpy.nanmin(data_plot[(data_plot.index < window[0]) & (data_plot.index > window[1])])
    ax.set(ylim=(min_value, max_value * 1.2))
    for col in cols:
        ax.plot(data_plot[col], color=colmap[col], label="%s" % col)
    ax.legend(fontsize="small")


def plot_proton(ax, H_data_plot, cols, H_scaling_window, colmap, ymax=None, proton_window=None):
    def extract_numeric(label):
        try:
            x = int(label.split("_")[0])
        except:
            x = label.split("_")[0]
        return x

    if ymax:
        max_scale_value = ymax
    else:
        max_scale_value = numpy.nanmax(H_data_plot[(H_data_plot.index > H_scaling_window[0]) &
                                                   (H_data_plot.index < H_scaling_window[1])])

    print("max value %s" % max_scale_value)
    print("H_scaling_range from %s to %s" % H_scaling_window)

    min_value = numpy.nanmin(H_data_plot)
    min_scale_value = min_value - (numpy.abs(max_scale_value) * 0.1)

    colgroups = list(set([el.split("_")[0] for el in cols]))
    colgroups = natsort.natsorted(colgroups, reverse=False, key=extract_numeric)

    rep_numbers = list()
    for colgroup_id in colgroups:
        colgroup = [el for el in cols if colgroup_id == el.split("_")[0]]
        rep_numbers.append(len(colgroup))

    nspecs = len(colgroups)
    replicate_offset = max_scale_value * 0.3
    conc_offset = 1.1 * max_scale_value
    max_offset = (nspecs - 1) * conc_offset + (max(rep_numbers) - 1) * replicate_offset
    print("max_offset %s" % max_offset)
    print("conc_offset %s" % conc_offset)
    print("nspecs %s" % nspecs)

    ax.plot()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set(ylabel="1H NMR signal [AU]")
    ax.set(xlabel="ppm")
    ax.set(ylim=(min_scale_value, 1.1 * (max_scale_value + max_offset)))

    if proton_window:
        ax.set(xlim=proton_window)
    ax.set_yticklabels([])
    ax.invert_xaxis()

    offset = 0
    for colgroup_id in colgroups:
        colgroup = [el for el in cols if colgroup_id == el.split("_")[0]]
        rep_count = 0
        for col in colgroup:
            color = [colmap[el] for el in colmap if el.split("_")[0] == col.split("_")[0]
                     and el.split("_")[1][0:-1] == col.split("_")[1][0:-1]][0]
            mask = numpy.isfinite(H_data_plot[col].values)
            ax.plot(H_data_plot[col].keys()[mask] - rep_count * 0.1,
                    H_data_plot[col].values[mask] + offset + rep_count * replicate_offset,
                    color=color,
                    label="%s uM, #%s" % tuple(col.split("_")))
            rep_count = rep_count + 1
        offset = offset + conc_offset

    ax.legend(loc=1, fontsize="small", ncol=1, numpoints=3)


def plot_peaks_style_2(ax, data_plot, cols, window, exp_data, colmap, signal_type, ymax=None):
    def extract_numeric(label):
        try:
            x = int(label.split("_")[0])
        except:
            x = label.split("_")[0]
        return x

    ppm_type = {"reporter": "rep_ppm",
                "control": "ctrl_ppm"}
    value_type = {"reporter": "rep_value",
                  "control": "ctrl_value"}

    if ymax:
        max_value = ymax
    else:
        max_value = numpy.nanmax(data_plot[(data_plot.index < window[0]) & (data_plot.index > window[1])])

    min_value = numpy.nanmin(data_plot[(data_plot.index < window[0]) & (data_plot.index > window[1])])

    colgroups = list(set([el.split("_")[0] for el in cols]))
    colgroups = natsort.natsorted(colgroups, key=extract_numeric)
    npeaks = len(colgroups)
    max_offset = npeaks * 0.1

    ax.plot()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set(ylabel="19F %s signal [AU] @ %.2f ppm" % (signal_type, numpy.mean(window)))
    ax.set(xlim=(numpy.mean(window) + 0.05, numpy.mean(window) - max_offset))
    ax.set(ylim=(min_value, max_value * 1.2))
    ax.set_xticklabels([])

    offset = 0
    for colgroup_id in colgroups:
        colgroup = [el for el in cols if colgroup_id == el.split("_")[0]]
        for col in colgroup:
            picked_data = [(el[ppm_type[signal_type]], el[value_type[signal_type]]) for el in exp_data if
                           el["expnumber"] == col.split("_")[1]]

            ax.plot(data_plot[col].keys() - offset, data_plot[col].values,
                    color=colmap[col],
                    label="%s" % col)
            ax.plot(numpy.mean(window) - offset + 0 * numpy.arange(min_value, max_value / 2, 10000),
                    numpy.arange(min_value, max_value / 2, 10000), color="gray",
                    ls="dashed")
            if picked_data:
                picked_ppm, picked_value = picked_data[0]
                ax.plot(picked_ppm - offset, picked_value,
                        marker="x", markerfacecolor=colmap[col], markeredgecolor="black")

        offset = offset + 0.1

    ax.legend(loc=2, fontsize="small", ncol=3, numpoints=3)


def plot_NMR_data(org_Data, tgtdir, KD=150.0, cint=0.2,
                  proton_window=None, proton_scaling_window=None):
    def extract_numeric(label):
        try:
            x = int(label.split("_")[0])
        except:
            x = label.split("_")[0]
        return x

    Ki_lists = dict()
    for D2 in org_Data.keys():
        for TE in org_Data[D2].keys():
            for NS in org_Data[D2][TE].keys():
                for reporter in org_Data[D2][TE][NS].keys():
                    for rep_conc in org_Data[D2][TE][NS][reporter].keys():
                        for prot_conc in org_Data[D2][TE][NS][reporter][rep_conc].keys():

                            for ligand in org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"].keys():
                                data_plot = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "data_plot"]
                                H_data_plot = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "H_data_plot"]
                                colnames = data_plot.columns
                                refcols = [el for el in colnames if "R" in el]
                                datacols = [el for el in colnames]
                                datacols = natsort.natsorted(datacols, key=extract_numeric)
                                refcols = natsort.natsorted(refcols, key=extract_numeric)
                                rep_window = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "rep_window"]
                                ctrl_window = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "ctrl_window"]
                                ymax = numpy.nanmax(data_plot[refcols][(data_plot.index < rep_window[0]) & (
                                            data_plot.index > rep_window[1])])
                                protein = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["prot_name"]
                                prot_batchID = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["prot_batchID"]


                                fn1 = "%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    ligand, reporter, rep_conc, protein, prot_conc, TE, D2, NS)

                                # calculate values for reporter curves
                                Eo = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["Eo"]
                                Lo = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["Lo"]
                                r_values = {"KD": KD, "Eo": Eo, "Lo": Lo}

                                exp_data = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "exp_data"]

                                NMR_max_effect = org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["NMR_max_effect"]
                                lower_limit = NMR_max_effect + cint * (1 - NMR_max_effect)
                                upper_limit = 1 - cint * (1 - NMR_max_effect)

                                print("calculating Ki_list for %s" % ligand)
                                Ki_list, norm_factor = calc_Ki_value(r_values, exp_data, NMR_max_effect,
                                                                     lower_limit, upper_limit)

                                # add names to to Ki_list
                                for el in Ki_list:
                                    el.update({"protein": protein})
                                    el.update({"reporter": reporter})
                                    el.update({"ligand": ligand})

                                print("%s" % ligand)

                                othercolnames = [el for el in colnames if not (el in refcols)]
                                ncols = len(othercolnames)
                                colmap = dict()
                                cmap1 = plt.cm.gist_rainbow
                                for i in range(0, ncols):
                                    colmap[othercolnames[i]] = cmap1(i / float(ncols - 0.9))

                                ncols = len(refcols)
                                cmap2 = plt.cm.pink
                                for i in range(0, ncols):
                                    colmap[refcols[i]] = cmap2(i / float(ncols))

                                # plot ligand depiction
                                ligandimage = None
                                try:
                                    smiles = getSmiles(ligand)
                                    ligandimage = getDepiction(smiles)
                                    imageBox = ligandimage.getbbox()
                                    ligandimage = ligandimage.crop(imageBox)
                                    if ligandimage:
                                        figfilename = os.path.join(tgtdir, "%s_depiction.png" % fn1)
                                        print("writing %s" % figfilename)
                                        ligandimage.save(figfilename)

                                    else:
                                        print("no depiction for %s created" % ligand)
                                except:
                                    print("could not generate depiction for ligand %s with reporter %s" % (
                                        ligand, reporter))

                                # plot NMR signal refs
                                if ctrl_window:
                                    nplots = 3
                                else:
                                    nplots = 2

                                try:

                                    # plot all NMR signal data
                                    fig = plt.figure(figsize=(9, 3.54))
                                    ax_rep = plt.subplot2grid((2, nplots), (0, 1), colspan=1)
                                    ax_curve = plt.subplot2grid((2, nplots), (0, 0), colspan=1)
                                    ax_proton = plt.subplot2grid((2, nplots), (1, 0), colspan=nplots)
                                    ax_depiction = inset_axes(ax_proton, width=1.0, height=0.6, loc=2)

                                    figtitle = "Protein: %s (%s) (%s uM) \n" % (protein, prot_batchID, prot_conc)
                                    figtitle += "Reporter: %s (%s uM, KD: %s uM), " % (reporter, rep_conc, KD)
                                    figtitle += "at %sK, NS: %s, D2: %.0f ms \n" % (TE, NS, D2 * 1000)
                                    figtitle += "Ligand: %s " % ligand
                                    figtitle += "mean Ki: %s uM " % (Ki_list[-1]["mean Ki [uM]"])
                                    figtitle += "SD: %s " % (Ki_list[-1]["SD Ki [uM]"])

                                    fig.suptitle(figtitle, fontsize=7)

                                    if ctrl_window:
                                        ax_control = plt.subplot2grid((2, nplots), (0, 2), colspan=1)
                                        try:
                                            plot_peaks_style_2(ax_control, data_plot, datacols, ctrl_window,
                                                               exp_data, colmap, "control")
                                        except:
                                            print("could not plot control data with reporter %s with ligand %s" % (
                                                reporter, ligand))
                                    try:
                                        plot_peaks_style_2(ax_rep, data_plot, datacols, rep_window,
                                                           exp_data, colmap, "reporter", ymax)
                                    except:
                                        print("could not plot reporter data with reporter %s with ligand %s" % (
                                            reporter, ligand))
                                    try:
                                        plot_curves(ax_curve, ligand, reporter, protein, r_values, Ki_list, norm_factor,
                                                    colmap, lower_limit, upper_limit)
                                    except Exception as e:
                                        print("could not plot curve data with reporter %s with ligand %s" % (
                                            reporter, ligand))
                                        traceback.print_exc()

                                    try:
                                        if proton_scaling_window:
                                            H_scaling_window = proton_scaling_window
                                        else:
                                            H_scaling_window = (5, 11)

                                        plot_proton(ax_proton, H_data_plot, H_data_plot.columns,
                                                    H_scaling_window, colmap, proton_window=proton_window)
                                    except Exception as ex:
                                        print(ex)
                                        print("could not plot proton data with reporter %s with ligand %s" % (
                                            reporter, ligand))

                                    try:
                                        ax_depiction.tick_params(axis='x', which='both', bottom=False, top=False,
                                                                 labelbottom=False)
                                        ax_depiction.tick_params(axis='y', which='both', right=False, left=False,
                                                                 labelleft=False)
                                        for pos in ['right', 'top', 'bottom', 'left']:
                                            ax_depiction.spines[pos].set_visible(False)

                                        ax_depiction.imshow(ligandimage)
                                    except:
                                        print("could not insert depiction for ligand %s with reporter %s" % (
                                            ligand, reporter))

                                    figfilename = os.path.join(tgtdir, "%s_complete.png" % fn1)
                                    print("writing %s" % figfilename)

                                    plt.tight_layout(rect=[0, 0.0, 1, 0.90])
                                    fig.savefig(figfilename, dpi=300)
                                    fig.clear()
                                    plt.close("all")
                                    Ki_list[-1].update({"image": figfilename})
                                    Ki_list[-1].update({"comment": ", ".join([el["comment"] for el
                                                                              in Ki_list[0:-1]
                                                                              if not el["comment"] == ""])})

                                except Exception as e:
                                    print("could not plot data with reporter %s with ligand %s" % (
                                        reporter, ligand))
                                    traceback.print_exc()

                                Ki_lists.update({fn1: Ki_list})

    return Ki_lists


def return_peak_property_by_ppm(search_ppm, line, search_window_size=0.3, peak_property_name="peak_vol"):
    if search_ppm:
        peak_ppm = [el for el in line["peak_ppm"] if
                    (search_ppm + search_window_size / 2) > el > (search_ppm - search_window_size / 2)]
        if peak_ppm:
            if len(peak_ppm) == 1:
                peak_index = line["peak_ppm"].index(peak_ppm)
                peak_property = line[peak_property_name][peak_index]
                peak_ppm = peak_ppm[0]
            else:
                peak_property = numpy.max([line[peak_property_name][line["peak_ppm"].index(el)] for el in peak_ppm])
                peak_ppm = line["peak_ppm"][line[peak_property_name].index(peak_property)]
        else:
            peak_property = numpy.NaN
            peak_ppm = None

    return peak_ppm, peak_property


def subdivide_table(table, rep_ppm, ctrl_ppm,
                    rep_search_window_size=None,
                    ctrl_search_window_size=None, property_name="vol", ctrl_scaling=True):
    peak_property_name = "peak_%s" % property_name
    ctrl_property_name = "ctrl_%s" % property_name
    rep_property_name = "rep_%s" % property_name

    if rep_search_window_size:
        rep_search_halfwidth = rep_search_window_size / 2
    else:
        rep_search_halfwidth = 0.2

    if ctrl_search_window_size:
        ctrl_search_halfwidth = ctrl_search_window_size / 2
    else:
        ctrl_search_halfwidth = 0.2

    H_table = [el for el in table if el["NUC1"] == "1H"]
    F_table = [el for el in table if el["NUC1"] == "19F"]
    # subdivide table
    org_Data = dict()
    relax_times = set([el["D2"] for el in F_table])
    relax_times = relax_times - {0}

    for D2 in relax_times:
        org_Data[D2] = dict()
        subtableD2 = [el for el in F_table if el["D2"] == D2]
        temperatures = set([numpy.round(el["TE"]) for el in subtableD2])

        for TE in temperatures:
            org_Data[D2][TE] = dict()
            subtableTE = [el for el in subtableD2 if numpy.round(el["TE"]) == TE]
            samplingnumbers = set([el["NS"] for el in subtableTE])

            for NS in samplingnumbers:
                org_Data[D2][TE][NS] = dict()
                subtableNS = [el for el in subtableTE if el["NS"] == NS]
                reporters = set([el["rep_name"] for el in subtableNS])
                reporters = reporters - {"NA"}

                for reporter in reporters:
                    org_Data[D2][TE][NS][reporter] = dict()
                    subtableREP = [el for el in subtableNS if el["rep_name"] == reporter]
                    reporter_concentrations = set([el["rep_conc"] for el in subtableREP]) - {0, "NA"}

                    for rep_conc in reporter_concentrations:
                        org_Data[D2][TE][NS][reporter][rep_conc] = dict()
                        subtableREPC = [el for el in subtableREP if el["rep_conc"] == rep_conc]
                        prot_concentrations = set([el["prot_conc"] for el in subtableREPC]) - {0, "NA"}

                        for prot_conc in prot_concentrations:
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc] = dict()
                            subtablePCONC = [el for el in subtableREPC if el["prot_conc"] in [prot_conc, 0, "NA"]]
                            ligands = set([el["lig_name"] for el in subtablePCONC])
                            ligands = ligands - {"NA"}
                            rep_free_signal = [el for el in subtableREP
                                               if (el["lig_conc"] == "NA" or el["lig_conc"] == 0)
                                               and (el["prot_conc"] == "NA" or el["prot_conc"] == 0)
                                               and el["NUC1"] == "19F"]
                            rep_signal = [el for el in subtablePCONC
                                          if (el["lig_conc"] == "NA" or el["lig_conc"] == 0)
                                          and not (el["prot_conc"] == "NA" or el["prot_conc"] == 0)
                                          and el["NUC1"] == "19F"]

                            if len(rep_free_signal[0]["peak_ppm"]) == 2 and (rep_ppm or ctrl_ppm):
                                if rep_ppm or ctrl_ppm:
                                    if rep_ppm:
                                        print(
                                            "Two signals found in free reporter experiment and reporter ppm given. Second peak is used as control")
                                        peak_ppm = [el for el in rep_free_signal[0]["peak_ppm"] if
                                                    (rep_ppm + rep_search_halfwidth) > el > (
                                                                rep_ppm - rep_search_halfwidth)]
                                        if peak_ppm:
                                            peak_index = rep_free_signal[0]["peak_ppm"].index(peak_ppm)
                                            other_peak = {0, 1} - {peak_index}
                                            ctrl_ppm = rep_free_signal[0]["peak_ppm"][other_peak.pop()]
                                            print("control peak found at %.2f" % ctrl_ppm)
                                        else:
                                            sys.exit("No peak around %.2f ppm for reporter." % rep_ppm)
                                    elif ctrl_ppm:
                                        print(
                                            "Two signals found in free reporter experiment and control ppm given. Second peak is used as reporter")
                                        peak_ppm = [el for el in rep_free_signal[0]["peak_ppm"] if
                                                    (ctrl_ppm + ctrl_search_halfwidth) > el > (
                                                                ctrl_ppm - ctrl_search_halfwidth)]
                                        if peak_ppm:
                                            peak_index = rep_free_signal[0]["peak_ppm"].index(peak_ppm)
                                            other_peak = {0, 1} - {peak_index}
                                            rep_ppm = rep_free_signal[0]["peak_ppm"][other_peak.pop()]
                                            print("control peak found at %.2f" % ctrl_ppm)
                                        else:
                                            sys.exit("No peak around %.2f ppm for control." % rep_ppm)

                            elif len(rep_free_signal[0]["peak_ppm"]) == 2:
                                print(
                                    "Two signals found in free reporter experiment. Trying to determine control and reporter.")
                                signal_dev = dict()
                                for ppm in rep_free_signal[0]["peak_ppm"]:
                                    free_signal = numpy.mean(
                                        [return_peak_property_by_ppm(ppm, el,
                                                                     search_window_size=rep_search_halfwidth * 2,
                                                                     peak_property_name=peak_property_name)[1] for el in
                                         rep_free_signal])
                                    prot_signal = numpy.mean([return_peak_property_by_ppm(ppm, el,
                                                                                          search_window_size=rep_search_halfwidth * 2,
                                                                                          peak_property_name=peak_property_name)[
                                                                  1] for el in rep_signal])
                                    signal_dev.update({numpy.std([free_signal, prot_signal]): ppm})

                                if len(signal_dev) < 2:
                                    sys.exit(
                                        "Control and Reporter peak are too close. Please indicate one of them with the -r or -c option")

                                else:
                                    ctrl_ppm = signal_dev[sorted(signal_dev.keys())[0]]
                                    rep_ppm = signal_dev[sorted(signal_dev.keys())[1]]
                                    print("Reporter peak found at %.2f" % rep_ppm)
                                    print("Control peak found at %.2f" % ctrl_ppm)

                            elif len(rep_free_signal[0]["peak_ppm"]) == 1:
                                print("Found just one peak in free reporter experiment. No controls used.")
                                if rep_ppm:
                                    peak_ppm = [el for el in rep_free_signal[0]["peak_ppm"] if
                                                (rep_ppm + rep_search_halfwidth) > el > (
                                                            rep_ppm - rep_search_halfwidth)]
                                    if peak_ppm:
                                        peak_index = rep_free_signal[0]["peak_ppm"].index(peak_ppm)
                                        rep_ppm = rep_free_signal[0]["peak_ppm"][peak_index]
                                    else:
                                        sys.exit("No peak reporter peak around %.2f ppm." % rep_ppm)
                                else:
                                    max_peak = numpy.max(rep_free_signal[0]["peak_value"])
                                    peak_index = rep_free_signal[0]["peak_value"].index(max_peak)
                                    rep_ppm = rep_free_signal[0]["peak_ppm"][peak_index]

                            elif len(rep_free_signal[0]["peak_ppm"]) >= 2 and ctrl_ppm and rep_ppm:
                                print("%s peaks found in reference dataset" % len(rep_free_signal[0]["peak_ppm"]))
                                print("Control and Reporter ppm given.")

                                peak_ppm = [el for el in rep_free_signal[0]["peak_ppm"] if
                                            (rep_ppm + rep_search_halfwidth) > el > (rep_ppm - rep_search_halfwidth)]
                                if peak_ppm:
                                    peak_index = rep_free_signal[0]["peak_ppm"].index(peak_ppm)
                                    rep_ppm = rep_free_signal[0]["peak_ppm"][peak_index]
                                else:
                                    for peak in rep_free_signal[0]["peak_ppm"]:
                                        print("peak found in reference: %.2f" % peak)
                                    sys.exit("No reporter peak around %.2f ppm." % rep_ppm)

                                peak_ppm = [el for el in rep_free_signal[0]["peak_ppm"] if
                                            (ctrl_ppm + ctrl_search_halfwidth) > el > (
                                                        ctrl_ppm - ctrl_search_halfwidth)]

                                if peak_ppm:
                                    peak_index = rep_free_signal[0]["peak_ppm"].index(peak_ppm)
                                    ctrl_ppm = rep_free_signal[0]["peak_ppm"][peak_index]
                                else:
                                    for peak in rep_free_signal[0]["peak_ppm"]:
                                        print("peak found in reference: %.2f" % peak)
                                    sys.exit("No control peak around %.2f ppm." % ctrl_ppm)

                            else:
                                print("Too many peak in reference dataset. %s peaks found" % len(
                                    rep_free_signal[0]["peak_ppm"]))
                                for peak in rep_free_signal[0]["peak_ppm"]:
                                    print("peak found in reference: %s" % peak)
                                sys.exit(
                                    "Reporter and Control signal cannot be determined unambiguously. Please provide Control and Reporter ppm.")

                            if ctrl_ppm:
                                if False in [True for el in subtablePCONC if
                                             return_peak_property_by_ppm(ctrl_ppm, el,
                                                                         peak_property_name=peak_property_name)[
                                                 1] > 300000]:
                                    print("WARNING: Not all datasets contain control peak")

                            if False in [True for el in subtablePCONC if
                                         return_peak_property_by_ppm(rep_ppm, el,
                                                                     peak_property_name=peak_property_name)[
                                             1] > 300000]:
                                print("WARNING: Not all datasets contain reporter peak")

                            for line in rep_free_signal:
                                if rep_ppm:
                                    peak_ppm = [el for el in line["peak_ppm"] if
                                                (rep_ppm + rep_search_halfwidth) > el > (
                                                            rep_ppm - rep_search_halfwidth)]
                                    if peak_ppm:
                                        peak_index = line["peak_ppm"].index(peak_ppm)
                                        line["rep_ppm"] = line["peak_ppm"][peak_index]
                                        line["rep_value"] = line["peak_value"][peak_index]
                                        line["rep_vol"] = line["peak_vol"][peak_index]
                                        line["rep_width"] = line["peak_width"][peak_index]

                                    else:
                                        line["rep_ppm"] = numpy.NaN
                                        line["rep_value"] = numpy.NaN
                                        line["rep_vol"] = numpy.NaN
                                        line["rep_width"] = numpy.NaN

                                        print("No peak found for reporter around %f ppm for %s. Found these peaks %s "
                                              % (rep_ppm, line["name"], ["%.2f" % el for el in line["peak_ppm"]]))

                                if ctrl_ppm:
                                    peak_ppm = [el for el in line["peak_ppm"] if
                                                (ctrl_ppm + 0.15) > el > (ctrl_ppm - 0.15)]
                                    if peak_ppm:
                                        peak_index = line["peak_ppm"].index(peak_ppm)
                                        line["ctrl_ppm"] = line["peak_ppm"][peak_index]
                                        line["ctrl_value"] = line["peak_value"][peak_index]
                                        line["ctrl_vol"] = line["peak_vol"][peak_index]
                                        line["ctrl_width"] = line["peak_width"][peak_index]
                                    else:
                                        line["ctrl_ppm"] = numpy.NaN
                                        line["ctrl_value"] = numpy.NaN
                                        line["ctrl_vol"] = numpy.NaN
                                        line["ctrl_width"] = numpy.NaN

                                        print("No peak found for control around %f ppm for %s." % (
                                            ctrl_ppm, line["name"]))
                                else:
                                    line["ctrl_ppm"] = None
                                    line["ctrl_value"] = 1
                                    line["ctrl_vol"] = 1
                                    line["ctrl_width"] = None

                            for line in rep_signal:
                                if rep_ppm:
                                    peak_ppm = [el for el in line["peak_ppm"] if
                                                (rep_ppm + rep_search_halfwidth) > el > (
                                                            rep_ppm - rep_search_halfwidth)]
                                    if peak_ppm:
                                        peak_index = line["peak_ppm"].index(peak_ppm)
                                        line["rep_ppm"] = line["peak_ppm"][peak_index]
                                        line["rep_value"] = line["peak_value"][peak_index]
                                        line["rep_vol"] = line["peak_vol"][peak_index]
                                        line["rep_width"] = line["peak_width"][peak_index]

                                    else:
                                        line["rep_ppm"] = numpy.NaN
                                        line["rep_value"] = numpy.NaN
                                        line["rep_vol"] = numpy.NaN
                                        line["rep_width"] = numpy.NaN

                                        print("No peak found for reporter around %f ppm for %s. Found these peaks %s "
                                              % (rep_ppm, line["name"], ["%.2f" % el for el in line["peak_ppm"]]))

                                if ctrl_ppm:
                                    peak_ppm = [el for el in line["peak_ppm"] if
                                                (ctrl_ppm + ctrl_search_halfwidth) > el > (
                                                            ctrl_ppm - ctrl_search_halfwidth)]
                                    if peak_ppm:
                                        peak_index = line["peak_ppm"].index(peak_ppm)
                                        line["ctrl_ppm"] = line["peak_ppm"][peak_index]
                                        line["ctrl_value"] = line["peak_value"][peak_index]
                                        line["ctrl_vol"] = line["peak_vol"][peak_index]
                                        line["ctrl_width"] = line["peak_width"][peak_index]
                                    else:
                                        line["ctrl_ppm"] = numpy.NaN
                                        line["ctrl_value"] = numpy.NaN
                                        line["ctrl_vol"] = numpy.NaN
                                        line["ctrl_width"] = numpy.NaN

                                        print("No peak found for reporter around %f ppm for %s. Found these peaks %s "
                                              % (rep_ppm, line["name"], ["%.2f" % el for el in line["peak_ppm"]]))

                                else:
                                    line["ctrl_ppm"] = None
                                    line["ctrl_value"] = 1
                                    line["ctrl_vol"] = 1
                                    line["ctrl_width"] = None

                            if ctrl_scaling:
                                rep_fs = numpy.nanmean(
                                    [el[rep_property_name] / el[ctrl_property_name] for el in rep_free_signal])
                                rep_s = numpy.nanmean(
                                    [el[rep_property_name] / el[ctrl_property_name] for el in rep_signal])
                            else:
                                rep_fs = numpy.nanmean(
                                    [el[rep_property_name] / 1 for el in rep_free_signal])
                                rep_s = numpy.nanmean(
                                    [el[rep_property_name] / 1 for el in rep_signal])

                            rep_peak_width = numpy.abs(numpy.nanmedian([el["rep_width"] for el in rep_free_signal]))
                            rep_window_middle = numpy.nanmedian([el["rep_ppm"] for el in rep_free_signal])
                            if numpy.isnan(rep_window_middle):
                                rep_window_middle = rep_ppm
                                rep_peak_width = 0.01

                            if not rep_search_window_size:
                                rep_search_window_size = 20 * rep_peak_width

                            rep_search_window = (
                                rep_window_middle + rep_search_window_size / 2,
                                rep_window_middle - rep_search_window_size / 2)

                            if ctrl_ppm:
                                ctrl_peak_width = numpy.abs(
                                    numpy.nanmedian([el["ctrl_width"] for el in rep_free_signal]))
                                ctrl_window_middle = numpy.nanmedian([el["ctrl_ppm"] for el in rep_free_signal])

                                if not ctrl_search_window_size:
                                    ctrl_search_window_size = 20 * ctrl_peak_width

                                ctrl_search_window = (
                                    ctrl_window_middle + ctrl_search_window_size / 2,
                                    ctrl_window_middle - ctrl_search_window_size / 2)
                            else:
                                ctrl_search_window = None

                            Eo = prot_conc
                            Lo = rep_conc
                            prot_name = set([el["prot_name"] for el in rep_signal])
                            prot_name = (prot_name - {"NA"}).pop()

                            prot_batchID = set([el["prot_batchID"] for el in rep_signal])
                            prot_batchID = (prot_batchID - {"NA"}).pop()

                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["rep_free_signal"] = rep_free_signal
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["rep_signal"] = rep_signal
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["NMR_max_effect"] = rep_s / rep_fs
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["Eo"] = Eo
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["Lo"] = Lo
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["prot_name"] = prot_name
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["prot_batchID"] = prot_batchID

                            ref_plot = pandas.DataFrame()
                            for dataset in rep_free_signal:
                                name = "Rfree_%i" % int(dataset["expnumber"])
                                dataset["data"].columns = [name]
                                ref_plot = pandas.concat([ref_plot, dataset["data"]])

                            for dataset in rep_signal:
                                name = "0_R_%i" % int(dataset["expnumber"])
                                dataset["data"].columns = [name]
                                ref_plot = pandas.concat([ref_plot, dataset["data"]])

                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ref_plot"] = ref_plot
                            org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"] = dict()

                            for ligand in ligands:
                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand] = dict()
                                print("Searching peaks for ligand %s." % ligand)
                                dataList = [el for el in subtablePCONC if
                                            el["lig_name"] == ligand and el["NUC1"] == "19F"]
                                for line in dataList:
                                    if len(line["peak_ppm"]) > 1 and not ctrl_ppm:
                                        print("WARNING: more that one peak found for ligand %s in experiment %s" % (
                                            ligand, line["name"]))

                                    if rep_ppm:
                                        print("searching around %.2f ppm +/- %.2f for reporter peak."
                                              % (rep_window_middle, rep_search_window_size / 2))
                                        peak_ppm = return_peak_property_by_ppm(rep_window_middle, line,
                                                                               search_window_size=rep_search_window_size,
                                                                               peak_property_name=peak_property_name)[0]

                                        if peak_ppm:
                                            print("found reporter peak at %.2f." % peak_ppm)
                                            peak_index = line["peak_ppm"].index(peak_ppm)

                                            line["rep_ppm"] = line["peak_ppm"][peak_index]
                                            line["rep_value"] = line["peak_value"][peak_index]
                                            line["rep_vol"] = line["peak_vol"][peak_index]
                                            line["rep_width"] = line["peak_width"][peak_index]

                                        else:
                                            line["rep_ppm"] = numpy.NaN
                                            line["rep_value"] = numpy.NaN
                                            line["rep_vol"] = numpy.NaN
                                            line["rep_width"] = numpy.NaN

                                            print(
                                                "No peak found for reporter around %f ppm for %s. Found these peaks %s "
                                                % (rep_ppm, line["name"], ["%.2f" % el for el in line["peak_ppm"]]))

                                    if ctrl_ppm:
                                        print("searching around %.2f ppm +/- %.2f for control peak."
                                              % (ctrl_window_middle, ctrl_search_window_size / 2))
                                        peak_ppm = return_peak_property_by_ppm(ctrl_window_middle, line,
                                                                               search_window_size=ctrl_search_window_size,
                                                                               peak_property_name=peak_property_name)[0]

                                        if peak_ppm:
                                            print("found control peak at %.2f." % peak_ppm)

                                            peak_index = line["peak_ppm"].index(peak_ppm)

                                            line["ctrl_ppm"] = line["peak_ppm"][peak_index]
                                            line["ctrl_value"] = line["peak_value"][peak_index]
                                            line["ctrl_vol"] = line["peak_vol"][peak_index]
                                            line["ctrl_width"] = line["peak_width"][peak_index]

                                        else:
                                            line["ctrl_ppm"] = numpy.NaN
                                            line["ctrl_value"] = numpy.NaN
                                            line["ctrl_vol"] = numpy.NaN
                                            line["ctrl_width"] = numpy.NaN

                                            print("No peak found for control around %f ppm for %s." % (
                                                ctrl_ppm, line["name"]))
                                    else:
                                        line["ctrl_ppm"] = None
                                        line["ctrl_value"] = 1
                                        line["ctrl_vol"] = 1
                                        line["ctrl_width"] = None

                                rep_window = (
                                    rep_window_middle + 20 * rep_peak_width, rep_window_middle - 20 * rep_peak_width)
                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "dataList"] = dataList
                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "rep_window"] = rep_window

                                if ctrl_ppm:
                                    # ctrl_window_size = numpy.abs(numpy.nanmedian([el["ctrl_width"] for el in dataList]))
                                    # ctrl_window_middle = numpy.nanmedian([el["ctrl_ppm"] for el in dataList])
                                    ctrl_window = (
                                        ctrl_window_middle + 20 * ctrl_peak_width,
                                        ctrl_window_middle - 20 * ctrl_peak_width)
                                    org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                        "ctrl_window"] = ctrl_window
                                else:
                                    org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                        "ctrl_window"] = None

                                data_plot = pandas.DataFrame()
                                data_plot = pandas.concat([data_plot, ref_plot])
                                for dataset in dataList:
                                    name = "%3.i_%i" % (dataset["lig_conc"], int(dataset["expnumber"]))
                                    dataset["data"].columns = [name]
                                    data_plot = pandas.concat([data_plot, dataset["data"]])

                                # generate proton plots
                                # get proton data for ligand
                                H_ligand = [el for el in H_table
                                            if numpy.round(el["TE"]) == TE
                                            and el["rep_name"] == reporter
                                            and el["lig_name"] == ligand]

                                H_data_plot = pandas.DataFrame()
                                for dataset in H_ligand:
                                    name = "%3.i_%i" % (dataset["lig_conc"], int(dataset["expnumber"]))
                                    dataset["data"].columns = [name]
                                    H_data_plot = pandas.concat([H_data_plot, dataset["data"]])

                                if ctrl_scaling:
                                    exp_data = [{"expnumber": el["expnumber"],
                                                 "NMR_signal": el[rep_property_name] / el[ctrl_property_name] / rep_fs,
                                                 "rel_rep_signal": el[rep_property_name] / el[
                                                     ctrl_property_name] / rep_s,
                                                 "Io_value": el["lig_conc"],
                                                 "ctrl_ppm": el["ctrl_ppm"],
                                                 "ctrl_value": el["ctrl_value"],
                                                 "rep_ppm": el["rep_ppm"],
                                                 "rep_value": el["rep_value"]}
                                                for el in dataList]
                                else:
                                    exp_data = [{"expnumber": el["expnumber"],
                                                 "NMR_signal": el[rep_property_name] / 1 / rep_fs,
                                                 "rel_rep_signal": el[rep_property_name] / 1 / rep_s,
                                                 "Io_value": el["lig_conc"],
                                                 "ctrl_ppm": el["ctrl_ppm"],
                                                 "ctrl_value": el["ctrl_value"],
                                                 "rep_ppm": el["rep_ppm"],
                                                 "rep_value": el["rep_value"]}
                                                for el in dataList]

                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "exp_data"] = exp_data

                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "data_plot"] = data_plot

                                org_Data[D2][TE][NS][reporter][rep_conc][prot_conc]["ligands"][ligand][
                                    "H_data_plot"] = H_data_plot

    return org_Data, table


def write_NMR_table(table, table_filename):
    with open(table_filename, "w") as csvfile:
        fieldnames = ["name", "lig_conc", "lig_name", "rep_conc", "rep_name", "prot_conc", "prot_name", "prot_batchID",
                      "rep_ppm", "rep_width", "rep_vol", "rep_value",
                      "ctrl_ppm", "ctrl_width", "ctrl_vol", "ctrl_value",
                      "D2", "TE", "NS",
                      "EXP", "DATE"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for el in table:
            writer.writerow(el)

        csvfile.close()


def write_Ki_tables(Ki_lists, tgtdir):
    dialect = csv.excel_tab
    dialect.lineterminator = "\n"
    all_Ki_filename = os.path.join(tgtdir, "all_Ki_list.tsv")
    fieldnames = ['protein', 'reporter', 'ligand', 'expnumber',
                  'Io [uM]', 'Ki [uM]', 'NMR_signal [AU]', 'rel_rep_signal', 'KD', 'Lo [uM]', 'Eo [uM]', "confidence",
                  "exclude", "comment"]

    assay_results_filename = os.path.join(tgtdir, "assay_results.tsv")
    assay_result_fieldnames = ['ligand', 'protein', 'reporter', 'mean Ki [uM]', 'SD Ki [uM]', "# datapoints", "comment",
                               "db_comment", "image"]

    with open(all_Ki_filename, "w") as csvfile1:
        writer1 = csv.DictWriter(csvfile1, fieldnames=fieldnames, extrasaction="ignore", dialect=dialect)
        writer1.writeheader()

        with open(assay_results_filename, "w") as csvfile3:
            writer3 = csv.DictWriter(csvfile3, fieldnames=assay_result_fieldnames, extrasaction="ignore",
                                     dialect=dialect)
            writer3.writeheader()

            # for name in sorted(Ki_lists.keys()):
            for name in Ki_lists.keys():
                Ki_list = Ki_lists[name]

                for el in Ki_list[0:-1]:
                    writer1.writerow(el)

                writer3.writerow(Ki_list[-1])

                Ki_filename = os.path.join(tgtdir, "%s_Ki_list.tsv" % name)
                with open(Ki_filename, "w") as csvfile2:
                    writer2 = csv.DictWriter(csvfile2, fieldnames=fieldnames, extrasaction="ignore", dialect=dialect)
                    writer2.writeheader()
                    for el in Ki_list[0:-1]:
                        writer2.writerow(el)


def main():
    tgtdir = parameters.tgtdir
    srcdir = parameters.srcdir
    raw = parameters.raw
    cint = parameters.cint
    KD = parameters.KD
    rep_ppm = parameters.rep_ppm
    ctrl_ppm = parameters.ctrl_ppm
    property_name = parameters.property_name
    nref_sig = parameters.nref_sig
    reporter_search_window_size = parameters.reporter_search_window_size
    control_search_window_size = parameters.control_search_window_size
    proton_window = parameters.proton_window
    proton_scaling_window = parameters.proton_scaling_window
    ctrl_scaling = parameters.ctrl_scaling

    table = list()
    dataDirs = getDataDirs(srcdir)

    for dirname in dataDirs:
        status, line, dt = convert_bruker_nmrdata(dirname, raw=raw, nref_sig=nref_sig)
        if status:
            line.update({"data": dt})
            table.append(line)

    org_Data, table = subdivide_table(table, rep_ppm, ctrl_ppm,
                                      reporter_search_window_size,
                                      control_search_window_size, property_name=property_name,
                                      ctrl_scaling=ctrl_scaling)

    # remove non 19F data
    table = [el for el in table if el["NUC1"] == "19F"]
    table_filename = os.path.join(tgtdir, "NMR_peak_list.csv")
    write_NMR_table(table, table_filename=table_filename)

    Ki_lists = plot_NMR_data(org_Data, tgtdir, KD=KD, cint=cint,
                             proton_window=proton_window,
                             proton_scaling_window=proton_scaling_window)

    write_Ki_tables(Ki_lists, tgtdir)


if __name__ == '__main__':
    main()
