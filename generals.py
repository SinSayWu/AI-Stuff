import win32com.client as comclt
wsh = comclt.Dispatch("WScript.Shell")
wsh.AppActivate("Notepad") # select another application
wsh.SendKeys("a") # send the keys you want