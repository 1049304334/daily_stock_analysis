# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
===================================

设计模式：策略模式 (Strategy Pattern)
- BaseFetcher: 抽象基类，定义统一接口
- DataFetcherManager: 策略管理器，实现自动切换

防封禁策略：
1. 每个 Fetcher 内置流控逻辑
2. 失败自动切换到下一个数据源
3. 指数退避重试机制
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 配置日志
logger = logging.getLogger(__name__)


# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


class DataFetchError(Exception):
    """数据获取异常基类"""
    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""
    pass


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 实现通用的技术指标计算
    
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）

        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass

    def _fetch_field_data(self, stock_code: str, field: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取特定字段的数据（子类可选择实现）

        用于字段级回退机制，当某个字段缺失时，从当前数据源获取该字段

        Args:
            stock_code: 股票代码
            field: 字段名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含指定字段的数据DataFrame，失败返回None
        """
        # 默认实现：获取完整数据后提取指定字段
        try:
            df = self._fetch_raw_data(stock_code, start_date, end_date)
            if df is not None and not df.empty:
                # 标准化数据
                df = self._normalize_data(df, stock_code)
                # 只返回指定字段
                if field in df.columns:
                    return df[['date', field]] if 'date' in df.columns else df[[field]]
        except Exception:
            pass
        return None
    
    def get_daily_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）
            
        Returns:
            标准化的 DataFrame，包含技术指标
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")
        
        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)
            
            # Step 3: 数据清洗
            df = self._clean_data(df)
            
            # Step 4: 计算技术指标
            df = self._calculate_indicators(df)
            
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        计算指标：
        - MA5, MA10, MA20: 移动平均线
        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）
        """
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 量比：当日成交量 / 5日平均成交量
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # 保留2位小数
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    数据源策略管理器
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    
    切换策略：
    - 优先使用高优先级数据源
    - 失败后自动切换到下一个
    - 所有数据源都失败时抛出异常
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
    
    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表

        优先级动态调整逻辑：
        - 如果配置了 TUSHARE_TOKEN：Tushare 优先级提升为 0（最高）
        - 否则按默认优先级：
          0. EfinanceFetcher (Priority 0) - 最高优先级
          1. AkshareFetcher (Priority 1)
          2. PytdxFetcher (Priority 2) - 通达信
          2. TushareFetcher (Priority 2)
          3. BaostockFetcher (Priority 3)
          4. YfinanceFetcher (Priority 4)
        """
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher
        from src.config import get_config

        config = get_config()

        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）
        efinance = EfinanceFetcher()
        akshare = AkshareFetcher()
        tushare = TushareFetcher()  # 会根据 Token 配置自动调整优先级
        pytdx = PytdxFetcher()      # 通达信数据源
        baostock = BaostockFetcher()
        yfinance = YfinanceFetcher()

        # 初始化数据源列表
        self._fetchers = [
            efinance,
            akshare,
            tushare,
            pytdx,
            baostock,
            yfinance,
        ]

        # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 0）
        self._fetchers.sort(key=lambda f: f.priority)

        # 构建优先级说明
        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")
    
    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)
    
    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
        enable_field_fallback: bool = True
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源，支持字段级回退）

        故障切换策略：
        1. 从最高优先级数据源开始尝试
        2. 捕获异常后自动切换到下一个
        3. 记录每个数据源的失败原因
        4. 所有数据源失败后抛出详细异常

        字段级回退策略（当 enable_field_fallback=True）：
        1. 首选数据源获取主要数据
        2. 检测缺失字段
        3. 对缺失字段，使用其他数据源补全
        4. 合并所有字段数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            enable_field_fallback: 是否启用字段级回退

        Returns:
            Tuple[DataFrame, str]: (数据, 成功的数据源名称)

        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        errors = []
        primary_fetcher_success = None
        primary_df = None

        # 第一阶段：尝试获取主要数据
        for fetcher in self._fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )

                if df is not None and not df.empty:
                    primary_fetcher_success = fetcher
                    primary_df = df
                    logger.info(f"[{fetcher.name}] 主要数据获取成功")
                    break

            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # 继续尝试下一个数据源
                continue

        # 如果没有获取到主要数据，进行传统故障切换
        if primary_df is None:
            for fetcher in self._fetchers:
                try:
                    logger.info(f"故障切换：尝试 [{fetcher.name}] 获取 {stock_code}...")
                    df = fetcher.get_daily_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        days=days
                    )

                    if df is not None and not df.empty:
                        logger.info(f"[{fetcher.name}] 故障切换成功获取 {stock_code}")
                        return df, fetcher.name

                except Exception as e:
                    error_msg = f"[{fetcher.name}] 故障切换失败: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    continue

        # 第二阶段：字段级回退
        if primary_df is not None and enable_field_fallback:
            try:
                enhanced_df = self._apply_field_fallback(
                    primary_df, primary_fetcher_success, stock_code,
                    start_date, end_date, days
                )
                logger.info(f"[字段级回退] {stock_code} 最终数据完成，共 {len(enhanced_df)} 条记录")
                return enhanced_df, primary_fetcher_success.name + "_enhanced"
            except Exception as e:
                logger.warning(f"[字段级回退] {stock_code} 回退失败: {e}")
                # 回退失败，返回原始数据
                return primary_df, primary_fetcher_success.name

        # 第三阶段：返回原始数据或报错
        if primary_df is not None:
            return primary_df, primary_fetcher_success.name if primary_fetcher_success else "unknown"

        # 所有数据源都失败
        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)
    
    @property
    def available_fetchers(self) -> List[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]
    
    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        批量预取实时行情数据（在分析开始前调用）
        
        策略：
        1. 检查优先级中是否包含全量拉取数据源（efinance/akshare_em）
        2. 如果不包含，跳过预取（新浪/腾讯是单股票查询，无需预取）
        3. 如果自选股数量 >= 5 且使用全量数据源，则预取填充缓存
        
        这样做的好处：
        - 使用新浪/腾讯时：每只股票独立查询，无全量拉取问题
        - 使用 efinance/东财时：预取一次，后续缓存命中
        
        Args:
            stock_codes: 待分析的股票代码列表
            
        Returns:
            预取的股票数量（0 表示跳过预取）
        """
        from src.config import get_config
        
        config = get_config()
        
        # 如果实时行情被禁用，跳过预取
        if not config.enable_realtime_quote:
            logger.debug("[预取] 实时行情功能已禁用，跳过预取")
            return 0
        
        # 检查优先级中是否包含全量拉取数据源
        # 注意：新增全量接口（如 tushare_realtime）时需同步更新此列表
        # 全量接口特征：一次 API 调用拉取全市场 5000+ 股票数据
        priority = config.realtime_source_priority.lower()
        bulk_sources = ['efinance', 'akshare_em']  # TODO: 新增全量接口需同步更新此处
        
        # 如果优先级中前两个都不是全量数据源，跳过预取
        # 因为新浪/腾讯是单股票查询，不需要预取
        priority_list = [s.strip() for s in priority.split(',')]
        first_bulk_source_index = None
        for i, source in enumerate(priority_list):
            if source in bulk_sources:
                first_bulk_source_index = i
                break
        
        # 如果没有全量数据源，或者全量数据源排在第 3 位之后，跳过预取
        if first_bulk_source_index is None or first_bulk_source_index >= 2:
            logger.info(f"[预取] 当前优先级使用轻量级数据源(sina/tencent)，无需预取")
            return 0
        
        # 如果股票数量少于 5 个，不进行批量预取（逐个查询更高效）
        if len(stock_codes) < 5:
            logger.info(f"[预取] 股票数量 {len(stock_codes)} < 5，跳过批量预取")
            return 0
        
        logger.info(f"[预取] 开始批量预取实时行情，共 {len(stock_codes)} 只股票...")
        
        # 尝试通过 efinance 或 akshare 预取
        # 只需要调用一次 get_realtime_quote，缓存机制会自动拉取全市场数据
        try:
            # 用第一只股票触发全量拉取
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)
            
            if quote:
                logger.info(f"[预取] 批量预取完成，缓存已填充")
                return len(stock_codes)
            else:
                logger.warning(f"[预取] 批量预取失败，将使用逐个查询模式")
                return 0
                
        except Exception as e:
            logger.error(f"[预取] 批量预取异常: {e}")
            return 0
    
    def get_realtime_quote(self, stock_code: str):
        """
        获取实时行情数据（自动故障切换）
        
        故障切换策略（按配置的优先级）：
        1. EfinanceFetcher.get_realtime_quote()
        2. AkshareFetcher.get_realtime_quote(source="em")  - 东财
        3. AkshareFetcher.get_realtime_quote(source="sina") - 新浪
        4. AkshareFetcher.get_realtime_quote(source="tencent") - 腾讯
        5. 返回 None（降级兜底）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None
        """
        from .realtime_types import get_realtime_circuit_breaker
        from src.config import get_config
        
        config = get_config()
        
        # 如果实时行情功能被禁用，直接返回 None
        if not config.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None
        
        # 获取配置的数据源优先级
        source_priority = config.realtime_source_priority.split(',')
        
        errors = []
        
        for source in source_priority:
            source = source.strip().lower()
            
            try:
                quote = None
                
                if source == "efinance":
                    # 尝试 EfinanceFetcher
                    for fetcher in self._fetchers:
                        if fetcher.name == "EfinanceFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code)
                            break
                
                elif source == "akshare_em":
                    # 尝试 AkshareFetcher 东财数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="em")
                            break
                
                elif source == "akshare_sina":
                    # 尝试 AkshareFetcher 新浪数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="sina")
                            break
                
                elif source in ("tencent", "akshare_qq"):
                    # 尝试 AkshareFetcher 腾讯数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="tencent")
                            break
                
                if quote is not None and quote.has_basic_data():
                    logger.info(f"[实时行情] {stock_code} 成功获取 (来源: {source})")
                    return quote
                    
            except Exception as e:
                error_msg = f"[{source}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # 所有数据源都失败，返回 None（降级兜底）
        if errors:
            logger.warning(f"[实时行情] {stock_code} 所有数据源均失败，降级处理: {'; '.join(errors)}")
        else:
            logger.warning(f"[实时行情] {stock_code} 无可用数据源")
        
        return None
    
    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据（带熔断和降级）
        
        策略：
        1. 检查配置开关
        2. 检查熔断器状态
        3. 调用 AkshareFetcher.get_chip_distribution()
        4. 失败则返回 None（降级兜底）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            ChipDistribution 对象，失败则返回 None
        """
        from .realtime_types import get_chip_circuit_breaker
        from src.config import get_config
        
        config = get_config()
        
        # 如果筹码分布功能被禁用，直接返回 None
        if not config.enable_chip_distribution:
            logger.debug(f"[筹码分布] 功能已禁用，跳过 {stock_code}")
            return None
        
        # 检查熔断器状态
        circuit_breaker = get_chip_circuit_breaker()
        if not circuit_breaker.is_available("akshare_chip"):
            logger.warning(f"[熔断] 筹码接口处于熔断状态，跳过 {stock_code}")
            return None
        
        try:
            # 调用 AkshareFetcher 获取筹码分布
            for fetcher in self._fetchers:
                if fetcher.name == "AkshareFetcher":
                    if hasattr(fetcher, 'get_chip_distribution'):
                        chip = fetcher.get_chip_distribution(stock_code)
                        if chip is not None:
                            circuit_breaker.record_success("akshare_chip")
                            return chip
                    break
            
            return None
            
        except Exception as e:
            logger.error(f"[筹码分布] 获取 {stock_code} 失败: {e}")
            circuit_breaker.record_failure("akshare_chip", str(e))
            return None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票中文名称（自动切换数据源）
        
        尝试从多个数据源获取股票名称：
        1. 先从实时行情缓存中获取（如果有）
        2. 依次尝试各个数据源的 get_stock_name 方法
        3. 最后尝试让大模型通过搜索获取（需要外部调用）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票中文名称，所有数据源都失败则返回 None
        """
        # 1. 先检查缓存
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        
        # 初始化缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        # 2. 尝试从实时行情中获取（最快）
        quote = self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, 'name') and quote.name:
            name = quote.name
            self._stock_name_cache[stock_code] = name
            logger.info(f"[股票名称] 从实时行情获取: {stock_code} -> {name}")
            return name
        
        # 3. 依次尝试各个数据源
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_name'):
                try:
                    name = fetcher.get_stock_name(stock_code)
                    if name:
                        self._stock_name_cache[stock_code] = name
                        logger.info(f"[股票名称] 从 {fetcher.name} 获取: {stock_code} -> {name}")
                        return name
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 获取失败: {e}")
                    continue
        
        # 4. 所有数据源都失败
        logger.warning(f"[股票名称] 所有数据源都无法获取 {stock_code} 的名称")
        return None

    def _apply_field_fallback(
        self,
        primary_df: pd.DataFrame,
        primary_fetcher: BaseFetcher,
        stock_code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        days: int
    ) -> pd.DataFrame:
        """
        应用字段级回退机制

        Args:
            primary_df: 主要数据源获取的数据
            primary_fetcher: 主要数据源
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数

        Returns:
            补全后的DataFrame
        """
        df = primary_df.copy()

        # 检查哪些标准字段缺失
        standard_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        missing_fields = [col for col in standard_columns if col not in df.columns]

        if not missing_fields:
            logger.info(f"[字段级回退] {stock_code} 没有缺失字段")
            return df

        logger.info(f"[字段级回退] {stock_code} 缺失字段: {missing_fields}")

        # 按字段进行回退
        for field in missing_fields:
            logger.info(f"[字段级回退] 尝试补全字段: {field}")

            # 获取日期范围（用于字段回退）
            if start_date is None:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
                start_date = start_dt.strftime('%Y-%m-%d')

            # 尝试用其他数据源获取该字段
            for fetcher in self._fetchers:
                if fetcher == primary_fetcher:
                    continue  # 跳过主要数据源

                try:
                    # 获取该字段的数据
                    field_data = fetcher._fetch_field_data(stock_code, field, start_date, end_date)

                    if field_data is not None and not field_data.empty:
                        # 合并字段数据
                        if field in field_data.columns:
                            # 如果有日期列，按日期合并
                            if 'date' in field_data.columns:
                                # 确保主数据有日期列
                                if 'date' not in df.columns and 'date' in primary_df.columns:
                                    df = df.merge(
                                        field_data[['date', field]],
                                        on='date',
                                        how='left',
                                        suffixes=('', f'_{field}_fallback')
                                    )
                                else:
                                    # 如果没有日期列，直接添加
                                    df[field] = field_data[field].values
                            else:
                                # 没有日期列，直接添加
                                df[field] = field_data[field].values

                            logger.info(f"[字段级回退] {field} 字段补全成功 (来源: {fetcher.name})")
                            break

                except Exception as e:
                    logger.debug(f"[字段级回退] {fetcher.name} 获取字段 {field} 失败: {e}")
                    continue

            # 如果字段仍然缺失，尝试使用插值或其他方法
            if field not in df.columns:
                logger.warning(f"[字段级回退] {field} 字段所有数据源均无法获取，尝试计算或使用默认值")

                # 对数值型字段尝试插值或计算
                if field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    # 使用其他相关字段计算
                    if field == 'pct_chg' and 'close' in df.columns:
                        # 涨跌幅 = (今日收盘 - 昨日收盘) / 昨日收盘 * 100
                        df['pct_chg'] = df['close'].pct_change() * 100
                        logger.info(f"[字段级回退] pct_chg 已通过收盘价计算得出")

                # 如果是成交量字段，且收盘价有数据，可以估算
                elif field == 'volume' and 'close' in df.columns:
                    # 使用历史成交量的平均值估算
                    avg_volume = df['close'].rolling(window=5).mean()
                    df['volume'] = avg_volume.fillna(avg_volume.mean())
                    logger.info(f"[字段级回退] volume 已通过历史均值估算")

                # 特殊处理成交额字段：尝试通过成交量 × 平均价格计算
                elif field == 'amount':
                    logger.info(f"[字段级回退] 尝试计算成交额字段")

                    # 检查是否有成交量和价格数据可以用于计算
                    if 'volume' in df.columns and not df['volume'].isna().all():
                        # 尝试多种方式计算平均价格
                        avg_price = None

                        # 方式1：使用 (开盘 + 收盘 + 最高 + 最低) / 4
                        if all(col in df.columns for col in ['open', 'close', 'high', 'low']):
                            avg_price = (df['open'] + df['close'] + df['high'] + df['low']) / 4
                            logger.info(f"[字段级回退] 使用 OHLC 均价计算成交额")

                        # 方式2：使用 (开盘 + 收盘) / 2
                        elif all(col in df.columns for col in ['open', 'close']):
                            avg_price = (df['open'] + df['close']) / 2
                            logger.info(f"[字段级回退] 使用 开盘收盘均价 计算成交额")

                        # 方式3：仅使用收盘价
                        elif 'close' in df.columns:
                            avg_price = df['close']
                            logger.info(f"[字段级回退] 使用收盘价计算成交额")

                        # 计算成交额 = 成交量 × 平均价格
                        if avg_price is not None:
                            # 成交量单位通常是手(100股)，价格单位是元/股
                            # 成交额(元) = 成交量(手) × 100 × 平均价格(元/股)
                            df['amount'] = df['volume'] * 100 * avg_price
                            logger.info(f"[字段级回退] 成交额已通过计算得出: volume × 100 × avg_price")
                        else:
                            logger.warning(f"[字段级回退] 无法计算成交额：缺少价格数据")
                    else:
                        logger.warning(f"[字段级回退] 无法计算成交额：缺少成交量数据")

                # 添加默认值（仅当计算也失败时）
                if field not in df.columns:
                    if field in ['open', 'high', 'low', 'close']:
                        df[field] = df.get('close', 0)
                    elif field == 'volume':
                        df[field] = 0
                    elif field == 'amount':
                        df[field] = 0
                        logger.warning(f"[字段级回退] 成交额设置为0（所有数据源均无此字段且无法计算）")
                    elif field == 'pct_chg':
                        df[field] = 0.0

        # 最后一次清理：确保所有必要字段都存在
        for col in standard_columns:
            if col not in df.columns:
                logger.warning(f"[字段级回退] {col} 字段最终缺失，使用默认值")
                if col == 'date':
                    # 添加日期列（从索引生成）
                    df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
                elif col in ['open', 'high', 'low', 'close']:
                    df[col] = df.get('close', 0)
                elif col == 'volume':
                    df[col] = 0
                elif col == 'amount':
                    df[col] = 0
                elif col == 'pct_chg':
                    df[col] = 0.0

        # 重新排序列
        df = df[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg'] +
                [col for col in df.columns if col not in ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']]]

        # 清理数据
        df = df.dropna(subset=['close', 'volume'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        return df

    def batch_get_stock_names(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        批量获取股票中文名称

        先尝试从支持批量查询的数据源获取股票列表，
        然后再逐个查询缺失的股票名称。

        Args:
            stock_codes: 股票代码列表

        Returns:
            {股票代码: 股票名称} 字典
        """
        result = {}
        missing_codes = set(stock_codes)

        # 1. 先检查缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}

        for code in stock_codes:
            if code in self._stock_name_cache:
                result[code] = self._stock_name_cache[code]
                missing_codes.discard(code)

        if not missing_codes:
            return result

        # 2. 尝试批量获取股票列表
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_list') and missing_codes:
                try:
                    stock_list = fetcher.get_stock_list()
                    if stock_list is not None and not stock_list.empty:
                        for _, row in stock_list.iterrows():
                            code = row.get('code')
                            name = row.get('name')
                            if code and name:
                                self._stock_name_cache[code] = name
                                if code in missing_codes:
                                    result[code] = name
                                    missing_codes.discard(code)

                        if not missing_codes:
                            break

                        logger.info(f"[股票名称] 从 {fetcher.name} 批量获取完成，剩余 {len(missing_codes)} 个待查")
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 批量获取失败: {e}")
                    continue

        # 3. 逐个获取剩余的
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name
                missing_codes.discard(code)

        logger.info(f"[股票名称] 批量获取完成，成功 {len(result)}/{len(stock_codes)}")
        return result