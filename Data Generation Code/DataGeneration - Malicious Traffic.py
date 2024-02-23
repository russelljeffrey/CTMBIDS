import switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from datetime import datetime

class CollectTrainingStatsApp(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)
        self.initialize_file()

    def initialize_file(self):
        with open("FlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond,label\n')

    def append_to_file(self, data):
        with open("FlowStatsfile.csv", "a+") as file0:
            file0.write(data)

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    def request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        for stat in self.filter_priority_1_flows(body):
            flow_id, ip_src, tp_src, ip_dst, tp_dst, ip_proto = self.extract_flow_details(stat)
            icmp_code, icmp_type = self.extract_icmp_details(stat)
            packet_count_per_second, packet_count_per_nsecond = self.calculate_packet_rates(stat)
            byte_count_per_second, byte_count_per_nsecond = self.calculate_byte_rates(stat)
            self.append_to_file(self.format_data(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst, ip_proto, icmp_code, icmp_type, stat, packet_count_per_second, packet_count_per_nsecond, byte_count_per_second, byte_count_per_nsecond))

    def filter_priority_1_flows(self, body):
        return [flow for flow in body if flow.priority == 1]

    def extract_flow_details(self, stat):
        ip_src = stat.match['ipv4_src']
        ip_dst = stat.match['ipv4_dst']
        ip_proto = stat.match['ip_proto']
        if ip_proto == 1:
            icmp_code = stat.match.get('icmpv4_code', -1)
            icmp_type = stat.match.get('icmpv4_type', -1)
            tp_src = tp_dst = 0
        elif ip_proto == 6:
            icmp_code = icmp_type = -1
            tp_src = stat.match.get('tcp_src', 0)
            tp_dst = stat.match.get('tcp_dst', 0)
        elif ip_proto == 17:
            icmp_code = icmp_type = -1
            tp_src = stat.match.get('udp_src', 0)
            tp_dst = stat.match.get('udp_dst', 0)
        flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
        return flow_id, ip_src, tp_src, ip_dst, tp_dst, ip_proto

    def extract_icmp_details(self, stat):
        if stat.match['ip_proto'] == 1:
            icmp_code = stat.match.get('icmpv4_code', -1)
            icmp_type = stat.match.get('icmpv4_type', -1)
        else:
            icmp_code = icmp_type = -1
        return icmp_code, icmp_type

    def calculate_packet_rates(self, stat):
        try:
            packet_count_per_second = stat.packet_count / stat.duration_sec
            packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
        except ZeroDivisionError:
            packet_count_per_second = packet_count_per_nsecond = 0
        return packet_count_per_second, packet_count_per_nsecond

    def calculate_byte_rates(self, stat):
        try:
            byte_count_per_second = stat.byte_count / stat.duration_sec
            byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
        except ZeroDivisionError:
            byte_count_per_second = byte_count_per_nsecond = 0
        return byte_count_per_second, byte_count_per_nsecond

    def format_data(self, timestamp, datapath_id, flow_id, ip_src, tp_src, ip_dst, tp_dst, ip_proto, icmp_code, icmp_type, stat, packet_count_per_second, packet_count_per_nsecond, byte_count_per_second, byte_count_per_nsecond):
        return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            timestamp, datapath_id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
            ip_proto, icmp_code, icmp_type, stat.duration_sec, stat.duration_nsec,
            stat.idle_timeout, stat.hard_timeout, stat.flags, stat.packet_count,
            stat.byte_count, packet_count_per_second, packet_count_per_nsecond,
            byte_count_per_second, byte_count_per_nsecond, 1)

