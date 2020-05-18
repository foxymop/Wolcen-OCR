# import the necessary packages
import cv2
import time
import numpy as np
import xlsxwriter
import tkinter as tk
from tkinter import filedialog as fd
import pytesseract as pt
from pytesseract import Output



win = tk.Tk()
win.title("Wolcen OCR")

files = tk.filedialog.askopenfilenames(parent=win,initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("jpeg files","*.jpg"),("all files","*.*")))
start_time = time.time()
gear = win.tk.splitlist(files)





# TODO: .exe
# TODO: save the data to an Excel spreadsheet

# Note: OpenCV encodes by default in BGR

# BGR = (139, 178, 195) for Physical
# BGR = (122, 141, 186) for Rend
# BGR = ( 58, 123,  37) for Toxic
# BGR = ( 20,  55, 182) for Fire
# BGR = (203, 201, 160) for Frost
# BGR = (177, 140,  78) for Lightning
# BGR = (124, 192, 187) for Sacred
# BGR = (108,  32,  48) for Shadow
# BGR = (131,  35, 140) for Aether



# Weapon Damage on Attacks
def weapon_damage_on_attacks(image):

    types = ['Physical', 'Rend', 'Toxic', 'Fire', 'Frost', 'Lightning', 'Sacred', 'Shadow', 'Aether']
    dmg_on_attacks = []

    for type in types:

        #Define the BGR values for the accompanying damage type
        (B, G, R) = (0, 0, 0)
        if type == 'Physical':
            (B, G, R) = (139, 178, 195)
        elif type == 'Rend':
            (B, G, R) = (122, 141, 186)
        elif type == 'Toxic':
            (B, G, R) = ( 58, 123,  37)
        elif type == 'Fire':
            (B, G, R) = ( 20,  55, 182)
        elif type == 'Frost':
            (B, G, R) = (203, 201, 160)
        elif type == 'Lightning':
            (B, G, R) = (177, 140,  78)
        elif type == 'Sacred':
            (B, G, R) = (124, 192, 187)
        elif type == 'Shadow':
            (B, G, R) = (108, 32, 48)
        elif type == 'Aether':
            (B, G, R) = (131, 35, 140)

        # Threshold and erode the BGR image to find the colored damage text
        low_thresh = (max(0,B-15), max(0,G-15), max(0,R-15))
        high_thresh = (min(255,B+15), min(255,G+15), min(255,R+15))
        img_col = cv2.bitwise_not(cv2.inRange(image, low_thresh, high_thresh))

        # Define the kernel for the erosion operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if type == 'Frost' or type == 'Shadow':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        img_ero = cv2.erode(img_col, kernel)

        # Do some OCR on the color thresholded image to get the bounding box for the text
        d = pt.image_to_data(img_ero, config='digits', output_type=Output.DICT)
        (x, y, w, h) = (0, 0, 0, 0)
        for i in range(len(d['text'])):
            if d['text'][i].isnumeric():
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                break

        # Set up a mask to extract the gray data from the bounding box
        mask = np.zeros(img_col.shape, np.uint8)
        mask[y:y + h, x:x + w] = 255

        # Convert the image to gray
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Do some preprocessing to clean up the gray image
        eq1 = cv2.equalizeHist(img_gray)
        eq2 = cv2.inRange(eq1, (220), (255))

        # Get the final processed image
        eq3 = cv2.bitwise_not(cv2.bitwise_and(eq2, eq2, mask=mask))

        # Do some OCR on the processed image to find damage data
        dmg = [0, 0]
        d = pt.image_to_data(eq3, config='digits-', output_type=Output.DICT)
        for i in range(len(d['text'])):
            if d['text'][i]:
                dmg = d['text'][i].split('-')

        # Convert damage strings to ints
        for i in range(len(dmg)):
            dmg[i] = int(dmg[i])

        dmg.insert(0, type)
        dmg_on_attacks.append(dmg)


    return dmg_on_attacks


# Debug version to display processed images
def weapon_damage_on_attacks_debug(image, type):

    #Define the BGR values for the accompanying damage type
    (B, G, R) = (0, 0, 0)
    if type == 'Physical':
        (B, G, R) = (139, 178, 195)
    elif type == 'Rend':
        (B, G, R) = (122, 141, 186)
    elif type == 'Toxic':
        (B, G, R) = (58, 123, 37)
    elif type == 'Fire':
        (B, G, R) = (20, 55, 182)
    elif type == 'Frost':
        (B, G, R) = (203, 201, 160)
    elif type == 'Lightning':
        (B, G, R) = (177, 140, 78)
    elif type == 'Sacred':
        (B, G, R) = (124, 192, 187)
    elif type == 'Shadow':
        (B, G, R) = (108, 32, 48)
    elif type == 'Aether':
        (B, G, R) = (131, 35, 140)

    # Threshold and erode the BGR image to find the colored damage text
    low_thresh = (max(0,B-15), max(0,G-15), max(0,R-15))
    high_thresh = (min(255,B+15), min(255,G+15), min(255,R+15))
    img_col = cv2.bitwise_not(cv2.inRange(image, low_thresh, high_thresh))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if type == 'Frost' or type == 'Shadow':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    img_ero = cv2.erode(img_col, kernel)

    # Do some OCR on the color thresholded image to get the bounding box for the text
    d = pt.image_to_data(img_ero, config='digits', output_type=Output.DICT)
    print(d['text'])
    (x, y, w, h) = (0, 0, 0, 0)
    for i in range(len(d['text'])):
        if d['text'][i].isnumeric():
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img_ero, (x, y), (x+w, y+h), (0,0,0), 2)
            break

    cv2.imshow('img', img_ero)
    cv2.waitKey(0)

    # Set up a mask to extract the gray data from the bounding box
    mask = np.zeros(img_col.shape, np.uint8)
    mask[y:y + h, x:x + w] = 255

    # Convert the image to gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img', img_gray)
    cv2.waitKey(0)

    # Do some preprocessing to clean up the gray image
    eq1 = cv2.equalizeHist(img_gray)

    cv2.imshow('img', eq1)
    cv2.waitKey(0)

    eq2 = cv2.inRange(eq1, (220), (255))

    cv2.imshow('img', eq2)
    cv2.waitKey(0)

    # Get the final processed image
    eq3 = cv2.bitwise_not(cv2.bitwise_and(eq2, eq2, mask=mask))

    # Display the image
    cv2.imshow('img', eq3)
    cv2.waitKey(0)

    # Do some OCR on the processed image to find damage data
    dmg = [0, 0]
    d = pt.image_to_data(eq3, config='digits-', output_type=Output.DICT)
    print(d['text'])
    for i in range(len(d['text'])):
        if d['text'][i]:
            dmg = d['text'][i].split('-')

    # Convert damage strings to ints
    for i in range(len(dmg)):
        dmg[i] = int(dmg[i])

    return dmg

# Weapon Damage on Spells
def weapon_damage_on_spells(image):

    types = ['Physical', 'Rend', 'Toxic', 'Fire', 'Frost', 'Lightning', 'Sacred', 'Shadow', 'Aether']
    dmg_on_spells = []

    for type in types:

        blue_channel = image[:, :, 0]
        blue_inv = cv2.bitwise_not(blue_channel)

        #cv2.imshow('img', blue_inv)
        #cv2.waitKey(0)

        d = pt.image_to_data(blue_inv, config='--psm 1', output_type=Output.DICT)
        #print(d['text'])

        dmg = [0, 0]
        for i in range(len(d['text'])):
            if d['text'][i] == 'Spells' and d['text'][i-4] == type:
                dmg = d['text'][i-5].split('-')

        # Convert damage strings to ints
        for i in range(len(dmg)):
            dmg[i] = int(dmg[i])

        # Put the type at the start of the list
        dmg.insert(0, type)
        dmg_on_spells.append(dmg)

    return dmg_on_spells


# Accessory Damage
def accessory_damage(image):

    attack_damage = []
    spell_damage = []

    word = 'Attacks'
    types = ['Physical', 'Rend', 'Toxic', 'Fire', 'Frost', 'Lightning']

    blue_channel = image[:, :, 0]
    blue_inv = cv2.bitwise_not(blue_channel)

    d = pt.image_to_data(blue_inv, config='--psm 1', output_type=Output.DICT)

    for type in types:

        dmg = [0, 0]
        for i in range(len(d['text'])):
            if d['text'][i] == word and d['text'][i-4] == type:
                dmg = d['text'][i-5].split('-')

        # Convert damage strings to ints
        for i in range(len(dmg)):
            dmg[i] = int(dmg[i])

        # Put the type at the start of the list
        dmg.insert(0, type)
        attack_damage.append(dmg)

    word = 'Spells'
    types = ['Fire', 'Frost', 'Lightning', 'Sacred', 'Shadow', 'Aether']

    for type in types:

        dmg = [0, 0]
        for i in range(len(d['text'])):
            if d['text'][i] == word and d['text'][i-4] == type:
                dmg = d['text'][i-5].split('-')

        # Convert damage strings to ints
        for i in range(len(dmg)):
            dmg[i] = int(dmg[i])

        # Put the type at the start of the list
        dmg.insert(0, type)
        spell_damage.append(dmg)

    damage = [attack_damage, spell_damage]
    return damage


def offhand_damage(image):

    types = ['Fire', 'Frost', 'Lightning']
    dmg_on_spells = []

    blue_channel = image[:, :, 0]
    blue_inv = cv2.bitwise_not(blue_channel)

    d = pt.image_to_data(blue_inv, config='--psm 1', output_type=Output.DICT)

    for type in types:

        dmg = [0, 0]
        for i in range(len(d['text'])):
            if d['text'][i] == 'Spells' and d['text'][i - 4] == type:
                dmg = d['text'][i - 5].split('-')

        # Convert damage strings to ints
        for i in range(len(dmg)):
            dmg[i] = int(dmg[i])

        # Put the type at the start of the list
        dmg.insert(0, type)
        dmg_on_spells.append(dmg)

    return dmg_on_spells



# Find % Damage Modifier on any Item Type
def find_modifiers(image):

    dmg_mods = []
    types = ['Physical', 'Rend', 'Toxic', 'Fire', 'Frost', 'Lightning', 'Sacred', 'Shadow', 'Aether', 'Material', 'Elemental', 'Occult']

    blue_channel = image[:, :, 0]
    blue_inv = cv2.bitwise_not(blue_channel)

    d = pt.image_to_data(blue_inv, config='--psm 1', output_type=Output.DICT)

    for type in types:
        num = 0
        percent = '%'
        for i in range(len(d['text'])):
            if percent in d['text'][i] and d['text'][i+1] == type and d['text'][i+2] == 'Damage':
                num += int(d['text'][i].replace('%', '').replace('+', ''))

        dmg_mods.append([type, num])

    return dmg_mods

def determine_item_type(image):

    accessory_types = ['Accessory', 'Amulet', 'Belt', 'Ring']
    armor_types = ['Helmet', 'Spaulder', 'Gauntlet', 'Chest-piece', 'Pants', 'Boots']
    weapon_types = ['Axe', 'Battleaxe', 'Bow', 'Dagger', 'Greatsword', 'Mace', 'Pistol', 'Staff', 'Sword', 'Warhammer']
    offhand_types = ['Catalyst', 'Shield']

    blue_channel = image[:, :, 0]
    blue_inv = cv2.bitwise_not(blue_channel)

    d = pt.image_to_data(blue_inv, config='--psm 1', output_type=Output.DICT)

    # Determine the item type
    for accessory in accessory_types:
        if accessory in d['text']:
            return 'Accessory'
    for armor in armor_types:
        if armor in d['text']:
            return armor
    for offhand in offhand_types:
        if offhand in d['text']:
            return 'Offhand'
    for weapon in weapon_types:
        if weapon in d['text']:
            return 'Weapon'

    return ''




weapon_dmg_attacks = []
weapon_dmg_spells = []

offhand_dmg = []
offhand_mods = []

acc1_dmg_attacks = []
acc1_dmg_spells = []
acc1_mods = []

acc2_dmg_attacks = []
acc2_dmg_spells = []
acc2_mods = []

acc3_dmg_attacks = []
acc3_dmg_spells = []
acc3_mods = []

acc4_dmg_attacks = []
acc4_dmg_spells = []
acc4_mods = []

helm_mods = []

spaulder1_mods = []

spaulder2_mods = []

gauntlet1_mods = []

gauntlet2_mods = []

chest_mods = []

pants_mods = []

boots_mods = []

num_accessories = 0
num_spaulders = 0
num_gauntlets = 0

img = cv2.imread('Spaulder1.png')
mods = find_modifiers(img)
print(mods)

'''
for g in gear:

    img = cv2.imread(g)

    item_type = determine_item_type(img)
    
    if item_type == 'Weapon':
        weapon_dmg_attacks = weapon_damage_on_attacks(img)
        weapon_dmg_spells = weapon_damage_on_spells(img)
    elif item_type == 'Offhand':
        offhand_dmg = offhand_damage(img)
        offhand_mods = find_modifiers(img)
    elif item_type == 'Accessory':
        num_accessories += 1
        if num_accessories == 1:
            acc1_dmg_attacks = accessory_damage(img)[0]
            acc1_dmg_spells = accessory_damage(img)[1]
            acc1_mods = find_modifiers(img)
        if num_accessories == 2:
            acc2_dmg_attacks = accessory_damage(img)[0]
            acc2_dmg_spells = accessory_damage(img)[1]
            acc2_mods = find_modifiers(img)
        if num_accessories == 3:
            acc3_dmg_attacks = accessory_damage(img)[0]
            acc3_dmg_spells = accessory_damage(img)[1]
            acc3_mods = find_modifiers(img)
        if num_accessories == 4:
            acc4_dmg_attacks = accessory_damage(img)[0]
            acc4_dmg_spells = accessory_damage(img)[1]
            acc4_mods = find_modifiers(img)
    elif item_type == 'Gauntlet':
        num_gauntlets += 1
        if num_gauntlets == 1:
            gauntlet1_mods = find_modifiers(img)
        if num_gauntlets == 2:
            gauntlet2_mods = find_modifiers(img)
    elif item_type == 'Spaulder':
        num_spaulders += 1
        if num_gauntlets == 1:
            spaulder1_mods = find_modifiers(img)
        if num_gauntlets == 2:
            spaulder2_mods = find_modifiers(img)
    elif item_type == 'Helmet':
        helm_mods = find_modifiers(img)
    elif item_type == 'Chest-piece':
        chest_mods = find_modifiers(img)
    elif item_type == 'Pants':
        pants_mods = find_modifiers(img)
    elif item_type == 'Boots':
        boots_mods = find_modifiers(img)




workbook = xlsxwriter.Workbook('Gear.xlsx')


worksheet = workbook.add_worksheet('Weapon')
row = 0
col = 0
worksheet.write(row, col + 1, 'Attacks')

for i in weapon_dmg_attacks:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Spells')

for i in weapon_dmg_spells:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])



worksheet = workbook.add_worksheet('Offhand')

row = 0
col = 0
worksheet.write(row, col + 1, 'Spells')

for i in offhand_dmg:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Modifiers')

for i in offhand_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])



worksheet = workbook.add_worksheet('Accessory1')

row = 0
col = 0
worksheet.write(row, col + 1, 'Attacks')

for i in acc1_dmg_attacks:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, col + 1, 'Spells')

for i in acc1_dmg_spells:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Modifiers')

for i in acc1_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])



worksheet = workbook.add_worksheet('Accessory2')

row = 0
col = 0
worksheet.write(row, col + 1, 'Attacks')

for i in acc2_dmg_attacks:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, col + 1, 'Spells')

for i in acc2_dmg_spells:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Modifiers')

for i in acc2_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])



worksheet = workbook.add_worksheet('Accessory3')

row = 0
col = 0
worksheet.write(row, col + 1, 'Attacks')

for i in acc3_dmg_attacks:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, col + 1, 'Spells')

for i in acc3_dmg_spells:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Modifiers')

for i in acc3_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])




worksheet = workbook.add_worksheet('Accessory4')

row = 0
col = 0
worksheet.write(row, col + 1, 'Attacks')

for i in acc4_dmg_attacks:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, col + 1, 'Spells')

for i in acc4_dmg_spells:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])
    worksheet.write(row, col + 2, i[2])

row += 2
worksheet.write(row, 1, 'Modifiers')

for i in acc4_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])



worksheet = workbook.add_worksheet('Helmet')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in helm_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Spaulder1')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in spaulder1_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Spaulder2')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in spaulder2_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Gauntlet1')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in gauntlet1_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Gauntlet2')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in gauntlet2_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Chest-piece')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in chest_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Pants')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in pants_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])

worksheet = workbook.add_worksheet('Boots')
row = 0
col = 0
worksheet.write(row, 1, 'Modifiers')
for i in boots_mods:
    row += 1
    worksheet.write(row, col, i[0])
    worksheet.write(row, col + 1, i[1])







workbook.close()
'''

print('Done!')
print("--- %s seconds ---" % (time.time() - start_time))
win.mainloop()