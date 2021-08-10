import numpy as np

class GenToFile:

    def GenStr(self, data, HB, LB):
        start_code = "{:02x}".format(1)
        add_zeros = "{:02x}".format(0)
        str_HB = "{:02x}".format(HB)
        str_LB = "{:02x}".format(LB)
        str_data = "{:02x}".format(data)
        chk = self.ChecksumCalc([data, HB, LB, 1])
        string = ":" + start_code + str_HB + str_LB + add_zeros + str_data + chk


        return string
            
    def ChecksumCalc(self,data):
        s = 0
        for i in data:
            s = s+i
        checksum = s & 255
        checksum = checksum ^ 255
        checksum = checksum + 1
        chk = np.mod(checksum,256)
        return "{:02x}".format(chk)









