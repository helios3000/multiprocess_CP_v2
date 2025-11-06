import multiprocessing as mp
import threading
import serial
from queue import Queue


class SerialReceiver(mp.Process):

    def __init__(self, spo2_queue, ibp_queue, pump1_queue, pump2_queue, flow_queue, preload_queue, semaphore1):
        mp.Process.__init__(self)

        self.spo2_queue = spo2_queue
        self.ibp_queue = ibp_queue
        self.pump1_queue = pump1_queue
        self.pump2_queue = pump2_queue
        self.flow_queue = flow_queue
        self.preload_queue = preload_queue

        self.sema1 = semaphore1

    def run(self):

        ser = serial.Serial('COM4', 115200, timeout=0.001)  # mock: COM6
        print('serial connect success')
        print(ser)

        ser.write(b'\x80\x73\x73\x8f')  # 환자감시장치로 's'를 보내 데이터 송수신 시작

        print('Serial Receiver start')

        received_data = Queue()

        def serial_receiving():
            try:
                while True:
                    if ser.readable():
                        raw_packet = ser.readline()
                        # print(raw_packet)
                        received_data.put(raw_packet.hex())
            except KeyboardInterrupt:
                print("Serial Receiving Process interrupted by user.")
            except Exception as error:
                print("An error occurred in Serial Receiving Process:", str(error))

        def packet_processing():
            try:
                while True:

                    packet = received_data.get()

                    for i in range(0, len(packet), 2):
                        if packet[i:i + 2] == '80':
                            j_ref = 0
                            for j in range(i + 2, len(packet), 2):
                                if packet[j:j + 2] == '8f':

                                    spo2_h = packet[i + 6:i + 8]
                                    spo2_l = packet[i + 8:i + 10]

                                    spo2_val = (int(spo2_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                               (int(spo2_l, 16) & int('01111111', 2)) - 1000

                                    ibp_h = packet[i + 10:i + 12]
                                    ibp_l = packet[i + 12:i + 14]

                                    ibp_val = (int(ibp_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                              (int(ibp_l, 16) & int('01111111', 2)) - 512

                                    flow_h = packet[i + 18:i + 20]
                                    flow_l = packet[i + 20:i + 22]

                                    flow_val = (int(flow_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                               (int(flow_l, 16) & int('01111111', 2)) - 512

                                    sac_sig = packet[i + 22:i + 24]

                                    pump_sig = format(int(sac_sig, 16), 'b').zfill(8)

                                    pump1 = pump_sig[7]
                                    pump2 = pump_sig[6]

                                    preload_h = packet[i + 14:i + 16]
                                    preload_l = packet[i + 16:i + 18]

                                    preload_val = (int(preload_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                                  (int(preload_l, 16) & int('01111111', 2)) - 512

                                    self.spo2_queue.put(spo2_val)
                                    self.ibp_queue.put(ibp_val)
                                    self.pump1_queue.put(pump1)
                                    self.pump2_queue.put(pump2)
                                    self.flow_queue.put(flow_val)
                                    self.preload_queue.put(preload_val)

                                    self.sema1.release()

                                    j_ref = 1
                                    break

                            if j_ref == 0:
                                break

            except KeyboardInterrupt:
                print("Packet Processing Process interrupted by user.")
            except Exception as error:
                print("An error occurred in Packet Processing Process:", str(error))

        thread1 = threading.Thread(target=serial_receiving)
        thread2 = threading.Thread(target=packet_processing)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
