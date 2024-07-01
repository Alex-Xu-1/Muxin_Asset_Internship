'''
MACD技术指标交易策略：
macd金叉全仓买入并持有，macd死叉卖出并空仓。

策略使用步骤：
第一步：左侧策略代码编辑面板,代码第16-21行分别输入股票代码、指标参数、基准指数.
第二步：右上角选择回测日期、账户初始资金、回测频率.ps:本策略适合每日频率运行.
第三步：点击左上角“编译运行”白色按钮,进行快速回测,右边查看策略收益走势及运行日志.
第四步：点击右上角“进行回测”蓝色按钮,执行回测并进入回测详情页面,查看策略交易明
        细、历史持仓、策略风险指标等情况.
第五步：在回测详情页面,点击右上角“开启仿真交易”蓝色按钮,策略即可在实时行情中运行,
        打开交易信号提醒,在同花顺APP应用上实时收到策略交易信号.
'''
import talib
def init(context):   
    g.security = '600519.SH' #输入股票代码
    #设置MACD模型参数
    g.Short = 12 #短周期平滑均线参数
    g.Long = 26 #长周期平滑均线参数
    g.M = 9 #DIFF的平滑均线参数
    set_benchmark('000300.SH') #设置基准指数，默认为沪深300
    
def handle_bar(context,bar_dict): 
    macd = get_macd(g.security)
    if macd[-1]>0 and macd[-2]<0 and len(list(context.portfolio.stock_account.positions.keys())) == 0:
        order_value(g.security,context.portfolio.available_cash)    
        log.info("买入 %s" % (g.security))    
    if macd[-2]>0 and macd[-1]<0 and len(list(context.portfolio.stock_account.positions.keys())) > 0:    
        order_target(g.security,0)    
        log.info("卖出 %s" % (g.security))
def get_macd(stock):
    price = history(stock, ['close'], 500, '1d', True, 'pre', is_panel=1)['close']
    DIFF, DEA, MACD = talib.MACD(price.values,
            fastperiod = g.Short, slowperiod = g.Long, signalperiod = g.M)
    return MACD