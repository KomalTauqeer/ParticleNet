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
from constants import *

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)

norm_1Dhist = False
unnorm_1Dhist = True
 
def plot(ch):
    fileTT = TFile.Open("/ceph/ktauqeer/ULNtuples/UL17/TTCR/TTCR_TTToSemiLeptonic_jetchargetagger.root")
    fileData = TFile.Open("/ceph/ktauqeer/ULNtuples/UL17/TTCR/TTCR_SingleLepton_combined_jetchargetagger.root")

    treeTT = fileTT.Get("AnalysisTree")
    treeData = fileData.Get("AnalysisTree")

    # These things just have to be kept in memory so that ROOT does not make them disappear in the next loop
    pads = []
    paveTexts = []
    legends = []
    hist = {}

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
        if ch == "Muon":
            treeTT.Draw("jetchargetagger_WpWn >> TTWp(30, 0, 1)", "(true_scores_WpWn==1)*(n_mu==1)*(event_weight)")
            treeTT.Draw("jetchargetagger_WpWn >> TTWn(30, 0, 1)", "(true_scores_WpWn==0)*(n_mu==1)*(event_weight)", "same")
            for unc in uncertaintySourceMuon:
                treeTT.Draw("jetchargetagger_WpWn >> TTWp_%s_up(30, 0, 1)"%(unc),"(true_scores_WpWn==1)*(n_mu==1)" + "*(event_weight*" + unc + "_up/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWn_%s_up(30, 0, 1)"%(unc),"(true_scores_WpWn==0)*(n_mu==1)" + "*(event_weight*" + unc + "_up/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWp_%s_down(30, 0, 1)"%(unc),"(true_scores_WpWn==1)*(n_mu==1)" + "*(event_weight*" + unc + "_down/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWn_%s_down(30, 0, 1)"%(unc),"(true_scores_WpWn==0)*(n_mu==1)" + "*(event_weight*" + unc + "_down/" + nominalUncertaintyMuon[unc] + ")")
                hist["TTWp_"+unc+"_up"] = gDirectory.Get("TTWp_"+unc+"_up")
                hist["TTWn_"+unc+"_up"] = gDirectory.Get("TTWn_"+unc+"_up")
                hist["TTWp_"+unc+"_down"] = gDirectory.Get("TTWp_"+unc+"_down")
                hist["TTWn_"+unc+"_down"] = gDirectory.Get("TTWn_"+unc+"_down")

            treeData.Draw("jetchargetagger_WpWn >> DataWp(30, 0, 1)", "(true_scores_WpWn==1)*(n_mu==1)")
            treeData.Draw("jetchargetagger_WpWn >> DataWn(30, 0, 1)", "(true_scores_WpWn==0)*(n_mu==1)", "same")

        elif ch == "Electron":
            treeTT.Draw("jetchargetagger_WpWn >> TTWp(30, 0, 1)", "(true_scores_WpWn==1)*(n_ele==1)*(event_weight)")
            treeTT.Draw("jetchargetagger_WpWn >> TTWn(30, 0, 1)", "(true_scores_WpWn==0)*(n_ele==1)*(event_weight)", "same")
            for unc in uncertaintySourceElectron:
                treeTT.Draw("jetchargetagger_WpWn >> TTWp_%s_up(30, 0, 1)"%(unc),"(true_scores_WpWn==1)*(n_mu==1)" + "*(event_weight*" + unc + "_up/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWn_%s_up(30, 0, 1)"%(unc),"(true_scores_WpWn==0)*(n_mu==1)" + "*(event_weight*" + unc + "_up/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWp_%s_down(30, 0, 1)"%(unc),"(true_scores_WpWn==1)*(n_mu==1)" + "*(event_weight*" + unc + "_down/" + nominalUncertaintyMuon[unc] + ")")
                treeTT.Draw("jetchargetagger_WpWn >> TTWn_%s_down(30, 0, 1)"%(unc),"(true_scores_WpWn==0)*(n_mu==1)" + "*(event_weight*" + unc + "_down/" + nominalUncertaintyMuon[unc] + ")")
                hist["TTWp_"+unc+"_up"] = gDirectory.Get("TTWp_"+unc+"_up")
                hist["TTWn_"+unc+"_up"] = gDirectory.Get("TTWn_"+unc+"_up")
                hist["TTWp_"+unc+"_down"] = gDirectory.Get("TTWp_"+unc+"_down")
                hist["TTWn_"+unc+"_down"] = gDirectory.Get("TTWn_"+unc+"_down")
           
            treeData.Draw("jetchargetagger_WpWn >> DataWp(30, 0, 1)", "(true_scores_WpWn==1)*(n_ele==1)")
            treeData.Draw("jetchargetagger_WpWn >> DataWn(30, 0, 1)", "(true_scores_WpWn==0)*(n_ele==1)", "same")


    hist["TTWp"] = gDirectory.Get("TTWp")
    hist["TTWn"] = gDirectory.Get("TTWn")
    hist["DataWp"] = gDirectory.Get("DataWp")
    hist["DataWn"] = gDirectory.Get("DataWn")

    # Evaluating systematic uncertainty per bin and adding statistical at the end too
    nbin = 30
    errorGraphWp = TGraphAsymmErrors(nbin)
    errorGraphWn= TGraphAsymmErrors(nbin)

    for binItr in range(1,nbin+1):
        unc_Wp_up_sq = 0.
        unc_Wp_down_sq = 0.
        unc_Wn_up_sq = 0.
        unc_Wn_down_sq = 0.
        if (ch == "Muon"):
            for unc in uncertaintySourceMuon:
                unc_Wp_up_sq = unc_Wp_up_sq + (abs(hist["TTWp_"+unc+"_up"].GetBinContent(binItr)-hist["TTWp"].GetBinContent(binItr))**2)
                unc_Wp_down_sq = unc_Wp_down_sq + (abs(hist["TTWp"].GetBinContent(binItr)-hist["TTWp_"+unc+"_down"].GetBinContent(binItr))**2)
                unc_Wn_up_sq = unc_Wn_up_sq + (abs(hist["TTWn_"+unc+"_up"].GetBinContent(binItr)-hist["TTWn"].GetBinContent(binItr))**2)
                unc_Wn_down_sq = unc_Wn_down_sq + (abs(hist["TTWn"].GetBinContent(binItr)-hist["TTWn_"+unc+"_down"].GetBinContent(binItr))**2)
        elif (ch == "Electron"):
            for unc in uncertaintySourceElectron:
                unc_Wp_up_sq = unc_Wp_up_sq + (abs(hist["TTWp_"+unc+"_up"].GetBinContent(binItr)-hist["TTWp"].GetBinContent(binItr))**2)
                unc_Wp_down_sq = unc_Wp_down_sq + (abs(hist["TTWp"].GetBinContent(binItr)-hist["TTWp_"+unc+"_down"].GetBinContent(binItr))**2)
                unc_Wn_up_sq = unc_Wn_up_sq + (abs(hist["TTWn_"+unc+"_up"].GetBinContent(binItr)-hist["TTWn"].GetBinContent(binItr))**2)
                unc_Wn_down_sq = unc_Wn_down_sq + (abs(hist["TTWn"].GetBinContent(binItr)-hist["TTWn_"+unc+"_down"].GetBinContent(binItr))**2)
        unc_Wp_stat_sq = hist["TTWp"].GetBinError(binItr)**2 
        unc_Wn_stat_sq = hist["TTWn"].GetBinError(binItr)**2 
 
        unc_Wp_up_sq = unc_Wp_up_sq + unc_Wp_stat_sq
        unc_Wp_down_sq = unc_Wp_down_sq + unc_Wp_stat_sq
        unc_Wn_up_sq = unc_Wn_up_sq + unc_Wn_stat_sq
        unc_Wn_down_sq = unc_Wn_down_sq + unc_Wn_stat_sq
 
        errorGraphWp.SetPoint(binItr, hist["TTWp"].GetBinCenter(binItr), hist["TTWp"].GetBinContent(binItr))
        errorGraphWp.SetPointEXhigh(binItr, hist["TTWp"].GetBinWidth(binItr)/2.)
        errorGraphWp.SetPointEXlow(binItr, hist["TTWp"].GetBinWidth(binItr)/2.)
        errorGraphWp.SetPointEYhigh(binItr, math.sqrt(unc_Wp_up_sq))
        errorGraphWp.SetPointEYlow(binItr, math.sqrt(unc_Wp_down_sq))

        errorGraphWn.SetPoint(binItr, hist["TTWn"].GetBinCenter(binItr), hist["TTWn"].GetBinContent(binItr))
        errorGraphWn.SetPointEXhigh(binItr, hist["TTWn"].GetBinWidth(binItr)/2.)
        errorGraphWn.SetPointEXlow(binItr, hist["TTWn"].GetBinWidth(binItr)/2.)
        errorGraphWn.SetPointEYhigh(binItr, math.sqrt(unc_Wn_up_sq))
        errorGraphWn.SetPointEYlow(binItr, math.sqrt(unc_Wn_down_sq))

    if (norm_1Dhist):
        hist["TTWp"].Scale(1/hist["TTWp"].Integral())
        hist["TTWn"].Scale(1/hist["TTWn"].Integral())
        hist["DataWp"].Scale(1/hist["DataWp"].Integral())
        hist["DataWn"].Scale(1/hist["DataWn"].Integral())

    hist["TTWp"].SetLineColor(kRed)
    hist["TTWn"].SetLineColor(kBlue)
    hist["DataWp"].SetMarkerColor(kRed)
    hist["DataWn"].SetMarkerColor(kBlue)

    hist["TTWp"].SetLineWidth(3)
    hist["TTWn"].SetLineWidth(3)
    hist["DataWp"].SetMarkerStyle(20)
    hist["DataWn"].SetMarkerStyle(24)

    errorGraphWp.SetLineColor(kBlack)
    errorGraphWp.SetFillColor(kBlack)
    errorGraphWp.SetFillStyle(3244)
    errorGraphWn.SetLineColor(kBlack)
    errorGraphWn.SetFillColor(kBlack)
    errorGraphWn.SetFillStyle(3244)

    #hist["TTWp"].GetXaxis().SetTitle("Jet charge tagger output score")
    hist["TTWp"].GetXaxis().SetTitleSize(0.04)
    #hist["TTWp"].GetXaxis().SetLabelSize(0.04)
    hist["TTWp"].GetXaxis().SetLabelSize(0.)
    if (norm_1Dhist): hist["TTWp"].GetYaxis().SetTitle("Normalized to unity")
    if (unnorm_1Dhist): hist["TTWp"].GetYaxis().SetTitle("Events / (0.1)")
    hist["TTWp"].GetYaxis().SetTitleSize(0.05)
    #hist["TTWp"].GetYaxis().SetTitleSize(0.04)
    hist["TTWp"].GetYaxis().SetTitleOffset(0.98)
    hist["TTWp"].SetMaximum(hist["DataWp"].GetMaximum()*1.5)
    #hist["TTWp"].SetMaximum(hist["TTWp"].GetMaximum()*1.5)
    TGaxis.SetMaxDigits(3)
 
    hist["TTWp"].Draw("HIST")
    hist["TTWn"].Draw("HIST same")
    errorGraphWp.Draw("E2 SAME")
    errorGraphWn.Draw("E2 SAME")
    hist["DataWp"].Draw("same PEX0")
    hist["DataWn"].Draw("same PEX0")

    # Lumi text
    #if year == "UL16preVFP": CMSStyle.setCMSEra("UL2016_preVFP")
    #if year == "UL16postVFP": CMSStyle.setCMSEra("UL2016_postVFP")
    #if year == "UL17": CMSStyle.setCMSEra("UL2017")
    CMSStyle.setCMSEra("UL2017")
    CMSStyle.lumiTextSize = 0.
    #CMSStyle.lumiTextSize = 0.38
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
    pt.AddText("%s channel" %ch)
    pt.Draw("SAME")
    paveTexts.append(pt)

    # Legend
    leg = TLegend(0.3,0.7,0.8,0.8)
    #leg = TLegend(0.3,0.6,0.8,0.9)
    leg.SetNColumns(2)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetLineColor(0)
    leg.SetLineWidth(0)
    leg.SetLineStyle(0)
    leg.SetTextFont(43)
    leg.AddEntry(hist["TTWn"],"TT Simulation: W^{#minus}","L")
    leg.AddEntry(hist["TTWp"],"TT Simulation: W^{+}","L")
    leg.AddEntry(hist["DataWn"],"Data: W^{#minus}","lep")
    leg.AddEntry(hist["DataWp"],"Data: W^{+}","lep")
    leg.Draw()
    legends.append(leg)

    # Ratiopad 
    ratiopad.cd()
    ratiopad.SetTickx()
    ratiopad.SetTicky()
    # Making ratio graph        
    ratiogrWp = TGraphErrors(hist["TTWp"].GetNbinsX())
    for binItr in range(0,hist["TTWp"].GetNbinsX()+1):
                if(hist["TTWp"].GetBinContent(binItr) != 0 and hist["DataWp"].GetBinContent(binItr)!= 0):
                    ratiogrWp.SetPoint(binItr, hist["TTWp"].GetBinCenter(binItr), hist["DataWp"].GetBinContent(binItr)/hist["TTWp"].GetBinContent(binItr))
                    delta_ratio_Wp = (hist["DataWp"].GetBinContent(binItr)/hist["TTWp"].GetBinContent(binItr)) * math.sqrt((hist["DataWp"].GetBinError(binItr)/hist["DataWp"].GetBinContent(binItr))**2 + (hist["TTWp"].GetBinError(binItr)/hist["TTWp"].GetBinContent(binItr))**2)
                    ratiogrWp.SetPointError(binItr, 0, delta_ratio_Wp)

                else:
                    ratiogrWp.SetPoint(binItr, hist["TTWp"].GetBinCenter(binItr), 9999)

    ratiogrWn = TGraphErrors(hist["TTWn"].GetNbinsX())
    for binItr in range(0,hist["TTWn"].GetNbinsX()+1):
                if(hist["TTWn"].GetBinContent(binItr) != 0 and hist["DataWn"].GetBinContent(binItr)!= 0):
                    ratiogrWn.SetPoint(binItr, hist["TTWn"].GetBinCenter(binItr), hist["DataWn"].GetBinContent(binItr)/hist["TTWn"].GetBinContent(binItr))
                    delta_ratio_Wn = (hist["DataWn"].GetBinContent(binItr)/hist["TTWn"].GetBinContent(binItr)) * math.sqrt((hist["DataWn"].GetBinError(binItr)/hist["DataWn"].GetBinContent(binItr))**2 + (hist["TTWn"].GetBinError(binItr)/hist["TTWn"].GetBinContent(binItr))**2)
                    ratiogrWn.SetPointError(binItr, 0, delta_ratio_Wn)

                else:
                    ratiogrWn.SetPoint(binItr, hist["TTWn"].GetBinCenter(binItr), 9999)

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
    canvas.SaveAs('Jetchargetagger_WpWn_DataMC_UL17_sys.pdf')

    #raw_input("Drawn")
    pad.Delete()
    canvas.Delete()

plot("Muon")




