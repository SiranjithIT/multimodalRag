import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface userQuery{
  query: string;
}

export interface serverResponse{
  Response: string
}

@Injectable({
  providedIn: 'root'
})

export class RagService {
  constructor(private http: HttpClient){}
  url: string = "http://127.0.0.1:8000";

  uploadPDF(pdfFile: File): Observable<serverResponse>{
    const formData: FormData = new FormData();
    formData.append('pdf_file', pdfFile);
    return this.http.post<serverResponse>(`${this.url}/upload`, formData);
  }

  askQuery(query: userQuery): Observable<serverResponse>{
    return this.http.post<serverResponse>(`${this.url}/query/`,query);
  }

}
