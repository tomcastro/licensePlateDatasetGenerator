from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def numberToImage(lp, style='current'):

    # Standard size for Chilean LPs is 360x130
    canvas = Image.new('RGB', (1080, 390), 'white')

    if(style == 'current'):
        font = ImageFont.truetype('./fonts/cargo2.ttf', 200, encoding='unic')
    elif(style == 'old'):
        font = ImageFont.truetype('./fonts/helveticacond.otf',
                                  250, encoding='unic')

    draw = ImageDraw.Draw(canvas)
    draw.text((100, 50), lp, 'black', font)

    canvas.show()
