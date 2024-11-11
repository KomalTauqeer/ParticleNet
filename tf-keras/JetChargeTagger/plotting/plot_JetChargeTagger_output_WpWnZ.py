#Modified for DP note style
import os
import sys
import numpy as np
import optparse
import math as math
from ROOT import *
import CMSStyle
from copy import deepcopy

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", default= "UL17")
(options,args) = parser.parse_args()
year = options.year

norm_1Dhist = True
unnorm_1Dhist = False

def plot_multi_output(node):

    hist = {}
    node_string = {0: "Wp", 1: "Wn", 2: "Z"}

    inputFile = TFile.Open("../ternary_training/{}/WpWnZ_test.root".format(year))
    tree = inputFile.Get("AnalysisTree")

    # These things just have to be kept in memory so that ROOT does not make them disappear in the next loop
    pads = []
    paveTexts = []
    legends = []
    
    canvas = TCanvas("c1","c1",800,640)
    pad = TPad("pad","pad",0.,0.,1.,1.)
    pads.append(pad)
    
    canvas.cd()
    pad.Draw()
    pad.cd()
    pad.SetTickx()
    pad.SetTicky()
  
    if (norm_1Dhist or unnorm_1Dhist):
        tree.Draw("jetchargetagger_prob_node{} >> histWp(15, 0., 1.)".format(node_string[node]), "(jetchargetagger_true_ind==0)" )
        tree.Draw("jetchargetagger_prob_node{} >> histWn(15, 0., 1.)".format(node_string[node]), "(jetchargetagger_true_ind==1)" )
        tree.Draw("jetchargetagger_prob_node{} >> histZ(15, 0., 1.)".format(node_string[node]), "(jetchargetagger_true_ind==2)" )

    hist['Wp'] = gDirectory.Get("histWp")
    hist['Wn'] = gDirectory.Get("histWn")
    hist['Z'] = gDirectory.Get("histZ")

    if (norm_1Dhist):
        hist['Wp'].Scale(1/hist['Wp'].Integral())
        hist['Wn'].Scale(1/hist['Wn'].Integral())
        hist['Z'].Scale(1/hist['Z'].Integral())

    hist['Wp'].SetLineColor(kRed)
    hist['Wn'].SetLineColor(kBlue)
    hist['Z'].SetLineColor(kGreen+1)

    hist['Wp'].SetLineWidth(3)
    hist['Wn'].SetLineWidth(3)
    hist['Z'].SetLineWidth(3)

    hist['Wp'].SetLineStyle(7)
    hist['Wn'].SetLineStyle(8)
    hist['Z'].SetLineStyle(1)

    node_string_latex = {0: "W^{+}", 1: "W^{#minus}", 2: "Z"}
    binwidth = (hist['Z'].GetXaxis().GetXmax() - hist['Z'].GetXaxis().GetXmin())/hist['Z'].GetNbinsX()
    hist['Z'].GetXaxis().SetTitle("Jet charge tagger output score [{} node]".format(node_string_latex[node]))
    hist['Z'].GetXaxis().SetTitleSize(0.04)
    hist['Wn'].GetXaxis().SetLabelSize(0)
    if (norm_1Dhist): hist['Z'].GetYaxis().SetTitle("a.u.")
    if (unnorm_1Dhist): hist['Z'].GetYaxis().SetTitle("Events / %.2f" %binwidth)
    hist['Z'].GetYaxis().SetTitleSize(0.05)
    hist['Z'].GetYaxis().SetTitleOffset(0.8)
    hist['Z'].SetMaximum(hist['Z'].GetMaximum()*2.4) if node==2 else hist['Z'].SetMaximum(hist['Z'].GetMaximum()*1.8)
    TGaxis.SetMaxDigits(3)
 
    hist['Z'].Draw("HIST")
    hist['Wp'].Draw("HIST same")
    hist['Wn'].Draw("HIST same")

    # Lumi text
    if year == "UL16preVFP": CMSStyle.setCMSEra("UL2016_preVFP", show_lumi=False)
    if year == "UL16postVFP": CMSStyle.setCMSEra("UL2016_postVFP", show_lumi=False)
    if year == "UL17": CMSStyle.setCMSEra("UL2017", show_lumi=False)
    if year == "UL18": CMSStyle.setCMSEra("UL2018test", show_lumi=False)
    CMSStyle.lumiTextSize = 0.38
    CMSStyle.lumiTextOffset = 0.10
    CMSStyle.cmsTextSize = 0.75
    #CMSStyle.cmsTextOffset = 0.3 
    CMSStyle.writeExtraText = True
    CMSStyle.extraText = "Simulation Preliminary"
    CMSStyle.extraOverCmsTextSize = 0.74
    #CMSStyle.relPosX    = 0.05
    #CMSStyle.relPosY    = -0.065
    #CMSStyle.relExtraDX = 0.10
    #CMSStyle.relExtraDY = 0.30
    CMSStyle.setCMSLumiStyle(pad,1,outOfFrame=False)

    # Channel text
    #pt = TPaveText(0.125,0.58,0.35,0.7, "blNDC")
    #pt.SetFillStyle(0)
    #pt.SetBorderSize(0)
    #pt.SetTextAlign(13)
    #pt.SetTextSize(0.04)
    #pt.AddText("%s output node " %node_string_latex[node])
    #pt.Draw("SAME")
    #paveTexts.append(pt)

    # Legend
    #leg = TLegend(0.3,0.7,0.8,0.8)
    #leg = TLegend(0.4,0.65,0.85,0.85)
    leg = TLegend(0.75,0.6,0.9,0.8)
    #leg.SetNColumns(2)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetLineColor(0)
    leg.SetLineWidth(0)
    leg.SetLineStyle(0)
    leg.SetTextFont(43)
    leg.AddEntry(hist['Wp'],"W^{+}","L")
    leg.AddEntry(hist['Wn'],"W^{#minus}","L")
    leg.AddEntry(hist['Z'],"Z","L")
    leg.Draw()
    legends.append(leg)

    canvas.Update()

    plotpath = './TaggerOutputMulti/{}'.format(year)
    if not os.path.isdir(plotpath):
        os.makedirs(plotpath)

    canvas.SaveAs('{}/Jetchargetagger_output_WpWnZ_node{}.pdf'.format(plotpath, node))

    #raw_input("Drawn")
    pad.Delete()
    canvas.Delete()

def main():
    global year
    for node in [0,1,2]:
        plot_multi_output(node)

if __name__ == "__main__":
    main()





