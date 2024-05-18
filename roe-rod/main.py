# roe + rod(rate of dividend)
# roe = r
# rod = d (分红率)

# https://xueqiu.com/7316851490/107719720

r = roe = 0.25
d = rod = 0.30
y = year = 7

discount_rate = 1.1


net_estate = (1 + r * (1 - d)) ** y  # 净资产份额 每年去除分红的累乘
profit     = r * ((1 + r * (1 - d)) ** (y - 1)) # 去年的净资产 * roe
dividend    = d * profit / (discount_rate ** (y - 1) ) # 没年的利润进行分红 并折现
# dividend = r * d * (((1 + r * (1 - d)) / discount_rate) ** (y - 1))
# Sn = a1 * (1 - qn) / (1 - q) {an = a1 * q^{n-1}}
discounted_net_estate = ((1 + r * (1 - d)) / discount_rate) ** y
discounted_dividend_sum = r * d * (1 - ((1 + r * (1 - d)) / discount_rate) ** y) / (1 - ((1 + r * (1 - d)) / discount_rate))
eval_pb = discounted_net_estate + discounted_dividend_sum

print(eval_pb)




