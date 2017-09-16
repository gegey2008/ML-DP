" the call to :runtime you can find below.  If you wish to change any of those
" settings, you should do it in this file (/etc/vim/vimrc), since debian.vim
" will be overwritten everytime an upgrade of the vim packages is performed.
" It is recommended to make changes after sourcing debian.vim since it alters
" the value of the 'compatible' option.

" This line should not be removed as it ensures that various options are
" properly set to work with the Vim-related packages available in Debian.
runtime! debian.vim

" Uncomment the next line to make Vim more Vi-compatible
" NOTE: debian.vim sets 'nocompatible'.  Setting 'compatible' changes numerous
" options, so any other options should be set AFTER setting 'compatible'.
"set compatible

" Vim5 and later versions support syntax highlighting. Uncommenting the next
" line enables syntax highlighting by default.
if has("syntax")
  syntax on
endif

" If using a dark background within the editing area and syntax highlighting
" turn on this option as well
"set background=dark

" Uncomment the following to have Vim jump to the last position when
" reopening a file
"if has("autocmd")
"  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
"endif

" Uncomment the following to have Vim load indentation rules and plugins
" according to the detected filetype.
"if has("autocmd")
"  filetype plugin indent on
"endif

" The following are commented out as they cause vim to behave a lot
" differently from regular Vi. They are highly recommended though.
"set showcmd		" Show (partial) command in status line.
"set showmatch		" Show matching brackets.
"set ignorecase		" Do case insensitive matching
"set smartcase		" Do smart case matching
"set incsearch		" Incremental search
"set autowrite		" Automatically save before commands like :next and :make
"set hidden		" Hide buffers when they are abandoned
"set mouse=a		" Enable mouse usage (all modes)

" Source a global configuration file if available
if filereadable("/etc/vim/vimrc.local")
  source /etc/vim/vimrc.local
endif


"" set Vundle for vim
"set nocompatible              " required
"filetype off                  " required
"" set the runtime path to include Vundle and initialize
"set rtp+=~/.vim/bundle/Vundle.vim
"call vundle#begin()
"" alternatively, pass a path where Vundle should install plugins
"" call vundle#begin('~/some/path/here')
"" let Vundle manage Vundle, required
"" Add all your plugins here (note older versions of Vundle used Bundle instead of Plugin)
"
"" filesystem
"Plugin 'gmarik/Vundle.vim'
"Plugin 'jistr/vim-nerdtree-tabs'
"Plugin 'kien/ctrlp.vim'
"
"" html
"" isnowfy only compatible with python not python3
"Plugin 'isnowfy/python-vim-instant-markdown'
"Plugin 'jtratner/vim-flavored-markdown'
"Plugin 'suan/vim-instant-markdown'
"Plugin 'nelstrom/vim-markdown-preview'
"
"" python sytax checker
"Plugin 'nvie/vim-flake8'
"Plugin 'vim-scripts/Pydiction'
"Plugin 'vim-scripts/indentpython.vim'
"Plugin 'scrooloose/syntastic'
"
"" auto-completion stuff
"" Plugin 'klen/python-mode'
"" Plugin 'Valloric/YouCompleteMe'
"Plugin 'klen/rope-vim'
"Plugin 'davidhalter/jedi-vim'
"Plugin 'ervandew/supertab'
"" code folding
"Plugin 'tmhedberg/SimpylFold'
"
"" Colors!!!
"Plugin 'altercation/vim-colors-solarized'
"Plugin 'jnurmine/Zenburn'
"
"" All of your Plugins must be added before the following line
"call vundle#end()            " required
"
"filetype plugin indent on    " required
"let g:SimpylFold_docstring_preview = 1
"
"" autocomplete
""let g:ycm_autoclose_preview_window_after_completion=1
"
""custom keys
""let mapleader=" "
""map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>
""call togglebg#map("<F5>")
"
""colorscheme zenburn
""set guifont=Monaco:h14
"
"let NERDTreeIgnore=['\.pyc$', '\~$'] "ignore files in NERDTree
"
""I don't like swap files
"set noswapfile
"
""turn on numbering
"set nu
"
""------------Start Python PEP 8 stuff----------------
""" Number of spaces that a pre-existing tab is equal to.
"au BufRead,BufNewFile *py,*pyw,*.c,*.h set tabstop=4
"
""spaces for indents
"au BufRead,BufNewFile *.py,*pyw set shiftwidth=4
"au BufRead,BufNewFile *.py,*.pyw set expandtab
"au BufRead,BufNewFile *.py set softtabstop=4
"
"" Use the below highlight group when displaying bad whitespace is desired.
"highlight BadWhitespace ctermbg=red guibg=red
"
"" Display tabs at the beginning of a line in Python mode as bad.
"au BufRead,BufNewFile *.py,*.pyw match BadWhitespace /^\t\+/
"
"" Make trailing whitespace be flagged as bad.
"au BufRead,BufNewFile *.py,*.pyw,*.c,*.h match BadWhitespace /\s\+$/
"
"" Wrap text after a certain number of characters
"au BufRead,BufNewFile *.py,*.pyw, set textwidth=100
"
"" Use UNIX (\n) line endings.
"au BufNewFile *.py,*.pyw,*.c,*.h set fileformat=unix
"
"" For full syntax highlighting:
"let python_highlight_all=1
"syntax on
"
"" Keep indentation level from previous line:
"autocmd FileType python set autoindent
"
"" make backspaces more powerfull
"set backspace=indent,eol,start 
"
""Folding based on indentation:
"autocmd FileType python set foldmethod=indent
"
""use space to open folds
"nnoremap <space> za 
""----------Stop python PEP 8 stuff--------------
"
""js stuff"
"autocmd FileType javascript setlocal shiftwidth=2 tabstop=2

"changed by yangyang

set tabstop=4
set softtabstop=4
set shiftwidth=4
set textwidth=79
set expandtab
set autoindent
set fileformat=unix
set nobackup
" set cursorline
set ruler
set autoindent
set splitbelow
set splitright
" Enable folding
set foldmethod=indent
set foldlevel=99
" Enable folding with the spacebar
nnoremap <space> za



"auto add pyhton header --start  
autocmd BufNewFile *.py 0r ~/.vim/vim_template/vim_pyhton_header  
autocmd BufNewFile *.py ks|call FileName()|'s  
autocmd BufNewFile *.py ks|call CreatedTime()|'s  
				      
fun FileName()  
	if line("$") > 10  
		let l = 10
	else
		let l = line("$")  
	endif   
	exe "1," . l . "g/File Name:.*/s/File Name:.*/File Name: " .expand("%")
endfun   																	

fun CreatedTime()
	if line("$") > 10
		let l = 10
	else  
		let l = line("$")
	endif
	exe "1," . l . "g/Created Time:.*/s/Created Time:.*/Created Time: " .strftime("%Y-%m-%d %T")
endfun
"auto add python header --end 
