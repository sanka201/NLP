import aiohttp
import asyncio
import pyjuicenet

async def main():
  async with aiohttp.ClientSession() as session:
    api = pyjuicenet.Api('02483afd-a37b-44a5-8dad-b05a993ca05c', session)
    devices = await api.get_devices()
    charger = devices[0]
    await charger.update_state()
    print(charger.voltage) # 240
    await charger.set_override(True) # Charge the car now ignoring the schedule

asyncio.run(main())