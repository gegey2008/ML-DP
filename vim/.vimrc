set nocompatible              " be iMproved, required
filetype off                  " requied!

set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'gmarik/vundle'

"------------------
" python sytax checker
"------------------
Plugin 'nvie/vim-flake8'
Plugin 'vim-scripts/Pydiction'
Plugin 'vim-scripts/indentpython.vim'
Plugin 'davidhalter/jedi-vim'
Plugin 'Valloric/YouCompleteMe'
Plugin 'jiangmiao/auto-pairs'
"------------------
" Code Completions
"------------------
Plugin 'Shougo/neocomplcache'
Plugin 'mattn/emmet-vim'
Plugin 'Raimondi/delimitMate'
Plugin 'ervandew/supertab'
"" snippets
Plugin 'garbas/vim-snipmate'
Plugin 'honza/vim-snippets'

"" code folding
Plugin 'tmhedberg/SimpylFold'


"------ snipmate dependencies -------
Plugin 'MarcWeber/vim-addon-mw-utils'
Plugin 'tomtom/tlib_vim'

"-----------------
" Fast navigation
"-----------------
"Plugin 'edsono/vim-matchit'
Plugin 'Lokaltog/vim-easymotion'

"--------------
" Fast editing
"--------------
Plugin 'tpope/vim-surround'
Plugin 'scrooloose/nerdcommenter'
Plugin 'sjl/gundo.vim'
Plugin 'godlygeek/tabular'
Plugin 'nathanaelkane/vim-indent-guides'

"--------------
" IDE features
"--------------
Plugin 'scrooloose/nerdtree'
Plugin 'humiaozuzu/TabBar'
Plugin 'majutsushi/tagbar'
Plugin 'mileszs/ack.vim'
Plugin 'kien/ctrlp.vim'
Plugin 'tpope/vim-fugitive'
Plugin 'Lokaltog/vim-powerline'
Plugin 'scrooloose/syntastic'
Plugin 'bronson/vim-trailing-whitespace'

"-------------
" Other Utils
"-------------
" Plugin 'humiaozuzu/fcitx-status'
Plugin 'nvie/vim-togglemouse'

"----------------------------------------
" Syntax/Indent for language enhancement
"----------------------------------------
"------- web backend ---------
Plugin '2072/PHP-Indenting-for-VIm'
Plugin 'tpope/vim-rails'
Plugin 'lepture/vim-jinja'
Plugin 'digitaltoad/vim-jade'

"------- web frontend ----------
Plugin 'othree/html5.vim'
"Plugin 'tpope/vim-haml'
Plugin 'pangloss/vim-javascript'
Plugin 'kchmck/vim-coffee-script'
Plugin 'nono/jquery.vim'
Plugin 'groenewege/vim-less'
Plugin 'wavded/vim-stylus'
Plugin 'nono/vim-handlebars'

"------- markup language -------
Plugin 'tpope/vim-markdown'
" Plugin 'timcharper/textile.vim'

"------- Ruby --------
" Plugin 'tpope/vim-endwise'

"------- Go ----------
Plugin 'fatih/vim-go'

"------- FPs ------
Plugin 'kien/rainbow_parentheses.vim'
" Plugin 'wlangstroth/vim-racket'
" Plugin 'vim-scripts/VimClojure'
" Plugin 'rosstimson/scala-vim-support'

"--------------
" Color Schemes
"--------------
Plugin 'rickharris/vim-blackboard'
Plugin 'altercation/vim-colors-solarized'
Plugin 'rickharris/vim-monokai'
Plugin 'tpope/vim-vividchalk'
Plugin 'Lokaltog/vim-distinguished'
Plugin 'chriskempson/vim-tomorrow-theme'
Plugin 'fisadev/fisa-vim-colorscheme'

call vundle#end()            " required
filetype plugin indent on    " requiredn 'gmarik/vundle'

"YouComplteteMe\
set runtimepath+=~/.vim/bundle/YouCompleteMe
let g:ycm_collect_identifiers_from_tags_files = 1 "开启YCM基于标签引擎
let g:ycm_collect_identifiers_from_comments_and_strings = 1 " 注释与字符串中的内容也用于补全
let g:syntastic_ignore_files=[".*\.py$"]
let g:ycm_seed_identifiers_with_syntax = 1                  " 语法关键字补全
let g:ycm_complete_in_comments = 1
let g:ycm_confirm_extra_conf = 0
let g:ycm_key_list_select_completion = ['<c-n>', '<Down>']  " 映射按键, 没有这个会拦截掉tab, 导致其他插件的tab不能用
let g:ycm_key_list_previous_completion = ['<c-p>', '<Up>']
let g:ycm_complete_in_comments = 1                          " 在注释输入中也能补全
let g:ycm_complete_in_strings = 1                           " 在字符串输入中也能补全
let g:ycm_collect_identifiers_from_comments_and_strings = 1 " 注释和字符串中的文字也会被收入补全
let g:ycm_global_ycm_extra_conf='~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'
let g:ycm_show_diagnostics_ui = 0                           " 禁用语法检查
inoremap <expr> <CR> pumvisible() ? "\<C-y>" : "\<CR>" |
nnoremap <c-j> :YcmCompleter GoToDefinitionElseDeclaration<CR>|
let g:ycm_min_num_of_chars_for_completion= 2                "从第2个键入字符就开始罗列匹配项

"Pydiction
filetype plugin on
let g:pydiction_location ='~/.vim/bundle/Pydiction/complete-dict'

"changed by yangyang

set number
"set guioptions-=r
"set guioptions-=L
"set guioptions-=b
set guifont=Monaco:h13
syntax on
"let g:solarized_termcolors=256
"set background=dark        "设置背景色"
"set nowrap    "设置不折行"
"set fileformat=unix    "设置以unix的格式保存文件"
set cindent        "设置C样式的缩进格式"
set tabstop=4    "设置table长度"
set softtabstop=4
set shiftwidth=4        "同上"
set showmatch    "显示匹配的括号"
set scrolloff=5        "距离顶部和底部5行"
set autoindent
"set laststatus=2    "命令行为两行"
set encoding=utf-8      "文件编码"
"set backspace=2
"set mouse=a        "启用鼠标"
"set selection=exclusive
"set selectmode=mouse,key
set matchtime=5
set ignorecase        "忽略大小写"
set incsearch
set hlsearch        "高亮搜索项"
"set noexpandtab        "不允许扩展table"
set whichwrap+=<,>,h,l
set autoread
"set cursorline        "突出显示当前行"
"set cursorcolumn
" Enable folding with the spacebar
nnoremap <space> za


"auto add pyhton header --start
autocmd BufNewFile *.py 0r ~/.vim/vim_template/vim_python_header
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

