import axios from 'axios';
import * as XLSX from 'xlsx';
import {Item} from './model';

export async function readData(file: string): Promise<Item[]> {
  const res = await axios.get(file, {responseType: 'arraybuffer'});
  const book = XLSX.read(res.data);
  const name = book.SheetNames[0];
  const sheet = book.Sheets[name];
  return XLSX.utils.sheet_to_json<Item>(sheet);
}
