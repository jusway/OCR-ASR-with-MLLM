import oss2

# 获取所有支持的地域和Endpoint信息
regions = oss2.get_service_info()

# 打印所有地域和对应的外网Endpoint
for region in regions:
    print(f"Region: {region.region_id}, Endpoint: {region.endpoint}")