import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RagService } from './rag-service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('Multimodal RAG');
  constructor(private ragService: RagService){}
  selectedFile?: File;
  query: string|null = null;

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length) {
      this.selectedFile = input.files[0];
    }
  }

  onUpload() {
    if (this.selectedFile) {
      this.ragService.uploadPDF(this.selectedFile).subscribe(res => {
        console.log(res);
      });
    }
  }

  askQuery(){
    if(this.query?.trim()){
      this.ragService.askQuery({query: this.query}).subscribe(res=>{
        console.log(res);
      })
    }
  }


}
