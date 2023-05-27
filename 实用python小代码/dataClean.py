import xlwings as xw

app =xw.App(visible=True,add_book=False)
workbook = app.books.open("dataset.xlsx")
sheet1 = workbook.sheets('sheet1')## sheets_list = workbook.sheets(1)  ->sheer1 = workbook.sheets(1)
#print(sheets_list)
#print(type(sheets_list))
# select_value = '福建省'#筛选值
# qxs_excel = workbook.sheets.add(select_value)#新建一个福建省的表格存数据
# qxs_excel.range("A1:F1").value = ["ID","姓名","性别","年龄","家庭年收入(万)"]
# range_value_list = []
#
# ####自定义遍历函数
# def readrange(excel):
#     for i in range(2, 2000):
#         select_sheet_value = "E" + str(i)  # 单个表格字符串
#         str_sheet1 = "A" + str(i) + ":" + "F" + str(i)# 整行表格字符串
#         select_value_sheet = excel.range(select_sheet_value).value#取出这里面的东西Ei
#         # print(select_value_sheet)
#         #if select_value_sheet.find(sdelect_value) != -1:  # 这里设置搜索条件判断
#         # if select_value_sheet == select_value:
#         if select_value_sheet:#先判断是否为空，不为空再允许
#             if select_value in select_value_sheet:
#                 str_value_row = excel.range(str_sheet1).value
#                 range_value_list.append(str_value_row)
#
# for excel in sheets_list:
#     readrange(excel)
#
# flag = 1  # 因为计算机从 0 开始，0 行已经写入标题，所以这里是 1；如果有多行标题，根据实际情况设置
# for i in range_value_list:
#     flag += 1
#     # 整行表格字符串
#     str_sheet1 = "A" + str(flag) + ":" + "E" + str(flag)
#     qxs_excel.range(str_sheet1).value = i



