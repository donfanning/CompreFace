/*
 * Copyright (c) 2020 the original author or authors
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
import { HTTP_INTERCEPTORS, HttpClient, HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { FormBuilder, FormsModule } from '@angular/forms';
import { MatRadioModule } from '@angular/material/radio';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CreateDialogComponent } from 'src/app/features/create-dialog/create-dialog.component';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AlertComponent } from './features/alert/alert.component';
import { ErrorInterceptor } from './core/auth/error.inerceptor';
import { DeleteDialogComponent } from './features/delete-dialog/delete-dialog.component';
import { EditDialogComponent } from './features/edit-dialog/edit-dialog.component';
import { FooterModule } from './features/footer/footer.module';
import { SnackBarModule } from './features/snackbar/snackbar.module';
import { ToolBarModule } from './features/tool-bar/tool-bar.module';
import { AppStoreModule } from './store/app-store.module';
import { MainLayoutComponent } from './ui/main-layout/main-layout.component';
import { TranslateLoader, TranslateModule } from '@ngx-translate/core';
import { TranslateHttpLoader } from '@ngx-translate/http-loader';
import { BreadcrumbsModule } from './features/breadcrumbs/breadcrumbs.module';
import { BreadcrumbsContainerModule } from './features/breadcrumbs.container/breadcrumbs.container.module';
import { CommonModule } from '@angular/common';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatDialogModule } from '@angular/material/dialog';
import { MatTableModule } from '@angular/material/table';
import { MatMenuModule } from '@angular/material/menu';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

@NgModule({
  declarations: [AppComponent, MainLayoutComponent, CreateDialogComponent, EditDialogComponent, AlertComponent, DeleteDialogComponent],
  imports: [
    BrowserModule,
    CommonModule,
    MatToolbarModule,
    MatButtonModule,
    MatCardModule,
    MatInputModule,
    MatDialogModule,
    MatTableModule,
    MatMenuModule,
    MatIconModule,
    MatProgressSpinnerModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    FormsModule,
    ToolBarModule,
    FooterModule,
    AppStoreModule,
    HttpClientModule,
    SnackBarModule,
    MatRadioModule,
    BreadcrumbsModule,
    BreadcrumbsContainerModule,
    TranslateModule.forRoot({
      loader: {
        provide: TranslateLoader,
        useFactory: HttpLoaderFactory,
        deps: [HttpClient],
      },
    }),
  ],
  providers: [
    FormBuilder,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: ErrorInterceptor,
      multi: true,
    },
  ],
  bootstrap: [AppComponent],
  exports: [],
  entryComponents: [CreateDialogComponent, AlertComponent, EditDialogComponent, DeleteDialogComponent],
})
export class AppModule {}

export function HttpLoaderFactory(http: HttpClient) {
  return new TranslateHttpLoader(http);
}
