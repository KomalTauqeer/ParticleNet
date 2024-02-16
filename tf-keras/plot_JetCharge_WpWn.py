import sys
sys.path.append("/work/ktauqeer/uhh2-106X-v2/CMSSW_10_6_28/src/UHH2/SemiLeptonicVBS/plotting")
from ROOT import *
from array import array
import numpy as np
from optparse import OptionParser
import math as math
import os
import CMS_lumi, tdrstyle, CMSStyle
from matplotlib import pyplot as plt
#from efficiency import calculate_efficiency

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

norm_1Dhist = True
unnorm_1Dhist = False
 
def plot():
    fileTT = TFile.Open("TTCR_TT_UL17_wgts.root")
    fileData = TFile.Open("TTCR_SingleLepton_UL17_wgts.root")
    #fileTT = TFile.Open("TTCR_TT_UL18_wgts.root")
    #fileData = TFile.Open("TTCR_SingleLepton_UL18.root")

    treeTT = fileTT.Get("AnalysisTree")
    treeData = fileData.Get("AnalysisTree")

    # These things just have to be kept in memory so that ROOT does not make them disappear in the next loop
    pads = []
    paveTexts = []
    legends = []
    
    canvas = TCanvas("c1","c1",800,640)
    #pad = TPad("pad","pad",0.,0.,1.,1.)
    pad = TPad("pad","pad",0.,0.25,1.,1.)
    ratiopad = TPad("ratiopad","ratiopad",0.,0.,1.,0.25)
    pads.append(pad)
    pads.append(ratiopad)
    
    canvas.cd()
    pad.SetBottomMargin(0.03)
    pad.Draw()
    ratiopad.SetTopMargin(0.03)
    ratiopad.SetBottomMargin(0.3)
    ratiopad.Draw()

    pad.cd()
    pad.SetTickx()
    pad.SetTicky()
   
    if (norm_1Dhist or unnorm_1Dhist):
        treeTT.Draw("dnnscores >> histWp(30, 0, 1)", "(truelabels==1)*(event_weight)")
        treeTT.Draw("dnnscores >> histWn(30, 0, 1)", "(truelabels==0)*(event_weight)", "same")
        treeData.Draw("dnnscores >> histWpData(30, 0, 1)", "truelabels==1")
        treeData.Draw("dnnscores >> histWnData(30, 0, 1)", "truelabels==0", "same")

    histWp = gDirectory.Get("histWp")
    histWn = gDirectory.Get("histWn")
    histWpData = gDirectory.Get("histWpData")
    histWnData = gDirectory.Get("histWnData")

    if (norm_1Dhist):
        histWp.Scale(1/histWp.Integral())
        histWn.Scale(1/histWn.Integral())
        histWpData.Scale(1/histWpData.Integral())
        histWnData.Scale(1/histWnData.Integral())

    histWp.SetLineColor(kRed)
    histWn.SetLineColor(kBlue)
    histWpData.SetMarkerColor(kRed)
    histWnData.SetMarkerColor(kBlue)

    histWp.SetLineWidth(3)
    histWn.SetLineWidth(3)
    histWpData.SetMarkerStyle(20)
    histWnData.SetMarkerStyle(24)

    #histWp.GetXaxis().SetTitle("jet charge (#kappa = 0.5)")
    #histWp.GetXaxis().SetTitleSize(0.04)
    histWp.GetXaxis().SetLabelSize(0)
    if (norm_1Dhist): histWp.GetYaxis().SetTitle("Normalized to unity")
    if (unnorm_1Dhist): histWp.GetYaxis().SetTitle("Events / (0.1)")
    histWp.GetYaxis().SetTitleSize(0.05)
    histWp.GetYaxis().SetTitleOffset(0.7)
    histWp.SetMaximum(histWpData.GetMaximum()*1.5)
    TGaxis.SetMaxDigits(3)
 
    histWp.Draw("HIST")
    histWn.Draw("HIST same")
    histWpData.Draw("same PEX0")
    histWnData.Draw("same PEX0")

    # Lumi text
    #if year == "UL16preVFP": CMSStyle.setCMSEra("UL2016_preVFP")
    #if year == "UL16postVFP": CMSStyle.setCMSEra("UL2016_postVFP")
    #if year == "UL17": CMSStyle.setCMSEra("UL2017")
    CMSStyle.setCMSEra("UL2018")
    #CMSStyle.lumiTextSize = 0.
    CMSStyle.lumiTextSize = 0.38
    CMSStyle.lumiTextOffset = 0.10
    CMSStyle.cmsTextSize = 0.58
    #CMSStyle.cmsTextOffset = 0.3 
    CMSStyle.writeExtraText = True
    CMSStyle.extraText = "Work In Progress"
    CMSStyle.extraOverCmsTextSize = 0.6
    #CMSStyle.relPosX    = 0.05
    #CMSStyle.relPosY    = -0.065
    #CMSStyle.relExtraDX = 0.10
    #CMSStyle.relExtraDY = 0.30
    CMSStyle.setCMSLumiStyle(pad,0)

    # Channel text
    #crstring = {"TT":"t#bar{t} control", "WJets":"W+jets control", "VBS": "VBS signal"}
    pt = TPaveText(0.125,0.84,0.225,0.915, "blNDC")
    pt.SetFillStyle(0)
    pt.SetBorderSize(0)
    pt.SetTextAlign(13)
    pt.SetTextSize(0.04)
    #pt.AddText("%s channel" %channel + ", %s region" %crstring[cr])
    pt.Draw("SAME")
    paveTexts.append(pt)

    # Legend
    leg = TLegend(0.3,0.7,0.8,0.8)
    leg.SetNColumns(2)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetLineColor(0)
    leg.SetLineWidth(0)
    leg.SetLineStyle(0)
    leg.SetTextFont(43)
    leg.AddEntry(histWn,"TT Simulation: W^{#minus}","L")
    leg.AddEntry(histWp,"TT Simulation: W^{+}","L")
    leg.AddEntry(histWnData,"Data: W^{#minus}","lep")
    leg.AddEntry(histWpData,"Data: W^{+}","lep")
    leg.Draw()
    legends.append(leg)

    # Ratiopad 
    ratiopad.cd()
    ratiopad.SetTickx()
    ratiopad.SetTicky()
    # Making ratio graph        
    ratiogrWp = TGraphErrors(histWp.GetNbinsX())
    for binItr in range(0,histWp.GetNbinsX()+1):
                if(histWp.GetBinContent(binItr) != 0 and histWpData.GetBinContent(binItr)!= 0):
                    ratiogrWp.SetPoint(binItr, histWp.GetBinCenter(binItr), histWpData.GetBinContent(binItr)/histWp.GetBinContent(binItr))
                    delta_ratio_Wp = (histWpData.GetBinContent(binItr)/histWp.GetBinContent(binItr)) * math.sqrt((histWpData.GetBinError(binItr)/histWpData.GetBinContent(binItr))**2 + (histWp.GetBinError(binItr)/histWp.GetBinContent(binItr))**2)
                    ratiogrWp.SetPointError(binItr, 0, delta_ratio_Wp)

                else:
                    ratiogrWp.SetPoint(binItr, histWp.GetBinCenter(binItr), 9999)

    ratiogrWn = TGraphErrors(histWn.GetNbinsX())
    for binItr in range(0,histWn.GetNbinsX()+1):
                if(histWn.GetBinContent(binItr) != 0 and histWnData.GetBinContent(binItr)!= 0):
                    ratiogrWn.SetPoint(binItr, histWn.GetBinCenter(binItr), histWnData.GetBinContent(binItr)/histWn.GetBinContent(binItr))
                    delta_ratio_Wn = (histWnData.GetBinContent(binItr)/histWn.GetBinContent(binItr)) * math.sqrt((histWnData.GetBinError(binItr)/histWnData.GetBinContent(binItr))**2 + (histWn.GetBinError(binItr)/histWn.GetBinContent(binItr))**2)
                    ratiogrWn.SetPointError(binItr, 0, delta_ratio_Wn)

                else:
                    ratiogrWn.SetPoint(binItr, histWn.GetBinCenter(binItr), 9999)

    # ratioplot properties
    xtitlestring = "Jet charge tagger output score"
    ratiogrWp.GetXaxis().SetTitle("%s" %xtitlestring)
    ratiogrWp.GetXaxis().SetTitleSize(0.15)
    ratiogrWp.GetXaxis().SetTitleOffset(0.9)
    #ratiogrWp.GetYaxis().SetRangeUser(0.7,1.3)
    ratiogrWp.GetYaxis().SetRangeUser(0.,2.1)
    ratiogrWp.GetYaxis().SetNdivisions(202)
    ratiogrWp.GetXaxis().SetLabelSize(0.12)
    ratiogrWp.GetYaxis().SetLabelSize(0.12)
    ratiogrWp.GetXaxis().SetLabelOffset(0.02)
    ratiogrWp.GetXaxis().SetTickLength(0.1)
    ratiogrWp.GetYaxis().SetTickLength(0.035)
    ratiogrWp.GetYaxis().SetTitleSize(0.14)
    ratiogrWp.GetYaxis().SetTitleOffset(0.21)
    ratiogrWp.GetYaxis().CenterTitle(True)
    ratiogrWp.GetYaxis().SetTitle("Data / MC")
    #ratiogrWp.SetMarkerStyle(20)
    ratiogrWp.SetLineWidth(2)
    ratiogrWp.SetLineColor(kRed)
    ratiogrWn.SetLineWidth(2)
    ratiogrWn.SetLineColor(kBlue)
    ratiogrWp.Draw()
    ratiogrWn.Draw("SAME")
    ratioaxis = ratiogrWp.GetXaxis()
    xmin = 0
    xmax = 1
    ratioaxis.SetLimits(xmin,xmax)

    ## Straight line for ratioplot
    ratioline = TLine(xmin, 1., xmax, 1.)
    ratioline.SetLineWidth(2)
    ratioline.SetLineColor(kBlack)
    ratioline.Draw("SAME")
    
    canvas.Update()
    #canvas.SaveAs('Jetchargetagger_WpWn_DataMC_UL18_testset_wgts.pdf')
    canvas.SaveAs('Jetchargetagger_WpWn_DataMC_UL17_wgts_norm.pdf')

    #raw_input("Drawn")
    pad.Delete()
    canvas.Delete()

plot()




