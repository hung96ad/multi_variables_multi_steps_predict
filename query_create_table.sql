-- ----------------------------
-- Table structure for historical_predictions_multi_step
-- ----------------------------
DROP TABLE IF EXISTS `historical_predictions_multi_step`;
CREATE TABLE `historical_predictions_multi_step`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_coin` smallint(6) NULL DEFAULT NULL,
  `time_create` int(11) NULL DEFAULT NULL,
  `price_predict` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `price_actual` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `price_preidct_last` float NULL DEFAULT NULL,
  `price_predict_previous` float NULL DEFAULT NULL,
  `price_actual_last` float NULL DEFAULT NULL,
  `price_actual_previous` float NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `time_create`(`time_create`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;

-- ----------------------------
-- Table structure for historical_train_multi_step
-- ----------------------------
DROP TABLE IF EXISTS `historical_train_multi_step`;
CREATE TABLE `historical_train_multi_step`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `id_coin` smallint(6) NULL DEFAULT NULL,
  `time_create` int(11) NULL DEFAULT NULL,
  `price_test` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `price_predict` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `RMSE` float NULL DEFAULT NULL,
  `max_error` float NULL DEFAULT NULL,
  `openTime_last` bigint(20) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `time_create`(`time_create`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;

-- ----------------------------
-- Table structure for candlestick_data
-- ----------------------------
DROP TABLE IF EXISTS `candlestick_data`;
CREATE TABLE `candlestick_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `idCoin` smallint(6) DEFAULT NULL,
  `openTime` bigint(20) DEFAULT NULL,
  `open` decimal(20,10) DEFAULT NULL,
  `high` decimal(20,10) DEFAULT NULL,
  `low` decimal(20,10) DEFAULT NULL,
  `close` decimal(20,10) DEFAULT NULL,
  `volume` decimal(20,10) DEFAULT NULL,
  `closeTime` bigint(20) DEFAULT NULL,
  `quoteAssetVolume` decimal(20,10) DEFAULT NULL,
  `numberOfTrader` int(11) DEFAULT NULL,
  `takerBuyBaseAssetVolume` decimal(20,10) DEFAULT NULL,
  `takerBuyQuoteAssetVolume` decimal(20,10) DEFAULT NULL,
  `ignore` decimal(20,10) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `openTime` (`openTime`),
  KEY `idCoin` (`idCoin`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for coin_info
-- ----------------------------
DROP TABLE IF EXISTS `coin_info`;
CREATE TABLE `coin_info` (
  `id` smallint(6) NOT NULL AUTO_INCREMENT,
  `symbol` char(20) DEFAULT NULL,
  `minQty` decimal(15,10) DEFAULT NULL,
  `tickSize` decimal(15,10) DEFAULT NULL,
  `status` char(20) DEFAULT NULL,
  `baseAsset` char(10) DEFAULT NULL,
  `quoteAsset` char(10) DEFAULT NULL,
  `openTime_last` bigint(20) DEFAULT NULL,
  `max_error` float DEFAULT NULL,
  `RMSE` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `symbol` (`symbol`) USING HASH
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
