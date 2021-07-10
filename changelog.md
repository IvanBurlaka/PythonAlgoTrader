## [ftx]
### Added
- Size increment awareness.

### Changed
- Open and close positions on candle closes that produce triggers (instead of brick edges).
- Use market order for opening position if limit order doesn't fill.

### Removed
- Unused code for prices from file.
- Moving open position order on next brick in the same direction if previous order was not filled.

## [0.3.0]
### Added
- Move open position order on next brick in the same direction if previous order was not filled.
- Add timeouts to ftx calls.
- Use 'total' USD balance instead of 'free'.

### Removed
- Multiple bricks flag - problematic and not needed with new stop rules.

## [0.2.0] - 2021-07-02
### Added
- Stop loss when candle closes crosses last brick open.
- Stop loss when candle closes 3/4 ATR away from position trigger brick close.

### Changed
- Better log messages.

## [0.1.0] - 2021-06-18
### Added
- Simple algo implementation.