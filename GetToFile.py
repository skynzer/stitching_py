
class GenToFile:

    def GenStr(self, data, str_num, addr):
        start_code = hex(1)[3:]
        add_zeros = hex(0)[3:]
        str_addr = hex(addr)[5:]
        chk = self.ChecksumCalc([data, addr, start_code])
        string = ":" + start_code + str_addr + add_zeros + chk
            
    def ChecksumCalc(self,data):
        s = int(0,2)
        for i in data:
            temp = int(i, 2)
            s = s+temp
        checksum = s^(2**(len(bit_s)+1)-1) + 1
        return hex(checksum)[3:]









