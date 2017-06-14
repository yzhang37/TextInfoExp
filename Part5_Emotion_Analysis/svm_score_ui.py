# encoding: utf-8
import wx
import sys
import os
import cPickle
from gensim.models.word2vec import Word2Vec
reload(sys)
sys.setdefaultencoding('utf-8')

class SVMScore_Dialog(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, u'SVM Score Evaluating', size=(500,500))
        
        self.menubar = wx.MenuBar()
        # data menu
        self.mnu_data = wx.Menu()
        self.mnu_data_open = wx.NewId()
        self.mnu_data_exit = wx.NewId()
        self.mnu_data.Append(self.mnu_data_open, u'&Load cPickle\tCtrl+O', u'Load a svm data which is stored in a cPickle file.')
        self.Bind(wx.EVT_MENU, self.evt_mnu_data_load, id=self.mnu_data_open)
        self.mnu_data.AppendSeparator()
        self.mnu_data.Append(self.mnu_data_exit, u'E&xit\tAlt+F4', u'Quit the program.')
        self.Bind(wx.EVT_MENU, self.evt_mnu_data_quit, id=self.mnu_data_exit)
        self.menubar.Append(self.mnu_data, u'&Data')
        
        # load menu bar
        self.SetMenuBar(self.menubar)
        
        
        panel = wx.Panel(self, wx.ID_ANY)
        basicLabel = wx.StaticText(panel, wx.ID_ANY, u"Sentence:")
        mainText = wx.TextCtrl(panel, wx.ID_ANY, u"",  size=(175, -1))
        sizer = wx.FlexGridSizer(cols=2, hgap=10, vgap=10)
        sizer.AddMany([basicLabel, mainText])
        panel.SetSizer(sizer)
    
    def evt_mnu_data_load(self, evt):
        dlg = wx.FileDialog(self, u'Load data',os.getcwdu(),style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            pass
        dlg.Destroy()
        
    def evt_mnu_data_quit(self, evt):
        self.Close()
    
    
app=wx.App()
dialog = SVMScore_Dialog()
dialog.Show(True)
app.MainLoop()