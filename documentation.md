# Sattmal Documentation

Sattmal is a local-first desktop document app. Your documents are stored on your machine, encrypted with your master password, and AI features use your configured local Ollama models.

## First Setup

1. Create a master password.
2. Choose where encrypted documents should be stored.
3. Start Ollama locally.
4. Select your local chat model in Settings.
5. Select a vision model if you want OCR for images or scanned PDFs.

If you lose the master password, encrypted documents cannot be recovered.

## Local AI Models

Sattmal uses Ollama running on your computer. The default Ollama URL is:

```text
http://localhost:11434
```

Use Settings to choose:

- Chat/Analysis Model: used for analysis, translation, document chat, and generated responses.
- Vision Model: used for images and scanned PDFs.
- Custom Trained Model Name: used only by the experimental custom model tools.
- Languages: used by Analyse and Translate.

No cloud AI provider is configured in the app.

## Document Storage

Encrypted documents and signatures are stored in the folder selected in Settings. You can choose another drive or folder by changing Encrypted Document Storage.

When the storage location changes, Sattmal moves the existing uploads and signatures into the new storage folder when possible.

## Uploading Documents

The dashboard supports these file types:

- PDF
- DOCX
- TXT
- PNG
- JPG/JPEG
- CSV
- ZIP

Add a title, description, and category to make documents easier to find later. The dashboard table includes Category as a searchable column.

## Viewing And Downloading

Use View to preview supported files. Use Download to save a decrypted copy. Downloads are handled inside the app so image files do not replace the desktop WebView page.

## Signing PDFs

Open a PDF and choose Sign.

You can:

- Draw a signature.
- Store a drawn signature for reuse.
- Type your name and choose from generated signature styles.
- Store a selected typed signature for reuse.
- Add text, dates, and numbers.
- Place items on the PDF and save a signed copy.

Reusable signatures are encrypted in local storage.

## Analyse, Translate, Respond, And Chat

The Analyse screen provides four tools:

- Analyse: summarize obligations, risks, and recommendations.
- Translate: translate extracted document text.
- Respond: draft a response based on the document.
- Chat: ask direct questions about the document.

Example chat questions:

- What does this clause actually mean?
- Are there any penalties I should know about?
- What am I agreeing to in section four?
- What changed between my results from last year and this year?

For scanned documents or images, select a local vision model in Settings.

The language dropdowns come from Settings. Add one language per line to make it available in Analyse and Translate.

## Categories

Manage categories in Settings. Categories appear during upload and on the dashboard table.

## Experimental Custom Model Tools

Settings includes an Experimental Custom Model Tools section behind an Options button.

These tools are intentionally hidden from the main dashboard because they may be slow or unreliable depending on your hardware, model setup, and document set.

## Email Sending

Email sending uses the Gmail address and app password configured in Settings. Attached files are decrypted only for the outgoing email operation.

## Troubleshooting

If AI actions fail:

- Make sure Ollama is installed.
- Make sure Ollama is running.
- Confirm the Ollama URL in Settings.
- Select a local chat model.
- Select a local vision model for scanned PDFs and images.

If a document is missing:

- Confirm the configured storage folder is available.
- If you use an external drive, make sure it is mounted before starting the app.

If a download does not start:

- Try the Download action again from the dashboard or document preview.
- Check whether your desktop WebView or operating system blocked the save prompt.
