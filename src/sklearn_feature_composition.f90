! Dense composition contracts: sklearn 1.6.1 compose/_column_transformer.py
! and pipeline.py. BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md.
module sklearn_feature_composition
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string
  use sklearn_function_transformer, only: function_transformer
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler,maxabs_scaler,robust_scaler,simple_imputer, &
    one_hot_encoder,ordinal_encoder,knn_imputer,missing_indicator,power_transformer,quantile_transformer, &
    kbins_discretizer,spline_transformer
  use sklearn_decomposition, only: pca,truncated_svd
  use sklearn_feature_selection, only: variance_threshold
  use sklearn_ieee_math, only: ieee_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_is_negative
  implicit none
  private
  public :: column_transformer,feature_union,transformer_branch,column_selector,feature_name_formatter
  ! Internal adapters shared with sequential pipelines, not facade algorithms.
  public :: supported,fit_model,fit_transform_model,transform_model,model_names,input_names,integer_text,real_text
  abstract interface
    function feature_name_formatter(transformer_name,feature_name) result(name)
      character(*), intent(in) :: transformer_name,feature_name
      character(:), allocatable :: name
    end function
    subroutine column_selector(x,feature_names,indices)
      import dp,feature_string
      real(dp), intent(in) :: x(:,:)
      type(feature_string), intent(in) :: feature_names(:)
      integer, allocatable, intent(out) :: indices(:)
    end subroutine
  end interface
  type :: transformer_branch
    character(:), allocatable :: name
    character(16) :: action='transform'
    class(*), allocatable :: model
    integer, allocatable :: columns(:)
    logical, allocatable :: mask(:)
    type(feature_string), allocatable :: column_names(:)
    procedure(column_selector), pointer, nopass :: selector=>null()
    real(dp) :: weight=1
    integer :: output_start_=1,output_stop_=0
  end type
  type :: column_transformer
    type(transformer_branch), allocatable :: transformers(:),transformers_(:)
    type(transformer_branch) :: remainder=transformer_branch(action='drop')
    procedure(feature_name_formatter), pointer, nopass :: name_formatter=>null()
    type(feature_string), allocatable :: feature_names_in_(:)
    logical :: verbose_feature_names_out=.true.,sparse_output_=.false.
    integer :: n_features_in_=0,n_features_out_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>column_fit
    procedure :: fit_transform=>column_fit_transform
    procedure :: transform=>column_transform
    procedure :: get_feature_names_out=>column_names
    procedure :: named_transformer_index=>column_named_index
    procedure :: output_indices=>column_output_indices
  end type
  type :: feature_union
    type(transformer_branch), allocatable :: transformer_list(:)
    type(feature_string), allocatable :: feature_names_in_(:)
    logical :: verbose_feature_names_out=.true.
    integer :: n_features_in_=0,n_features_out_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>union_fit
    procedure :: fit_transform=>union_fit_transform
    procedure :: transform=>union_transform
    procedure :: get_feature_names_out=>union_names
  end type
  type :: matrix_block
    real(dp), allocatable :: values(:,:)
  end type
contains
  function integer_text(i) result(s)
    integer, intent(in) :: i
    character(:), allocatable :: s
    character(32) :: buffer
    write(buffer,'(i0)') i;s=trim(buffer)
  end function
  function real_text(x) result(s)
    ! Shortest round-trip decimal, with Python float exponent/decimal layout.
    real(dp), intent(in) :: x
    character(:), allocatable :: s,digits,mantissa,signpart,exponent_text
    character(64) :: buffer,fmt
    real(dp) :: back
    integer :: d,e,pos,point,ios
    if(ieee_is_nan(x)) then
      s='nan';return
    end if
    if(.not.ieee_is_finite(x)) then
      s='inf';if(x<0) s='-inf'
      return
    end if
    signpart='';if(ieee_is_negative(x)) signpart='-'
    if(x==0) then
      s=signpart//'0.0';return
    end if
    do d=1,17
      write(fmt,'(a,i0,a)') '(es26.',d-1,'e3)'
      write(buffer,fmt) abs(x)
      pos=index(buffer,'E');if(pos==0) pos=index(buffer,'e')
      read(buffer(pos+1:),*) e
      if(e>308) cycle
      if(e==308) then
        if(adjustl(buffer(:pos-1))>'1.7976931348623157') cycle
      end if
      read(buffer,*,iostat=ios) back
      if(ios==0) then
        if(back==abs(x)) exit
      end if
    end do
    buffer=adjustl(buffer);pos=index(buffer,'E');if(pos==0) pos=index(buffer,'e')
    read(buffer(pos+1:),*) e
    mantissa=trim(buffer(:pos-1));point=index(mantissa,'.')
    digits=mantissa(:point-1)//mantissa(point+1:)
    do while(len(digits)>1)
      if(digits(len(digits):)/='0') exit
      digits=digits(:len(digits)-1)
    end do
    if(e>=-4.and.e<16) then
      point=e+1
      if(point<=0) then
        s='0.'//repeat('0',-point)//digits
      else if(point>=len(digits)) then
        s=digits//repeat('0',point-len(digits))//'.0'
      else
        s=digits(:point)//'.'//digits(point+1:)
      end if
    else
      exponent_text=integer_text(abs(e))
      if(len(exponent_text)<2) exponent_text='0'//exponent_text
      s=digits(:1)
      if(len(digits)>1) s=s//'.'//digits(2:)
      if(e<0) then
        s=s//'e-'//exponent_text
      else
        s=s//'e+'//exponent_text
      end if
    end if
    s=signpart//s
  end function
  logical function same_name(a,b) result(same)
    character(*), intent(in) :: a,b
    same=len(a)==len(b)
    if(same) same=a==b
  end function
  function input_names(p,supplied,stored) result(names)
    integer, intent(in) :: p
    type(feature_string), optional, intent(in) :: supplied(:),stored(:)
    type(feature_string), allocatable :: names(:)
    integer :: i
    if(present(supplied)) then
      call require(size(supplied)==p,'composition feature name count mismatch')
      do i=1,p
        call require(allocated(supplied(i)%text),'composition feature name is unallocated')
      end do
      names=supplied
      if(present(stored)) then
        call require(size(stored)==p,'composition stored feature name count mismatch')
        do i=1,p
          call require(same_name(names(i)%text,stored(i)%text),'composition input feature names differ')
        end do
      end if
    else if(present(stored)) then
      names=stored
    else
      allocate(names(p))
      do i=1,p
        names(i)%text='x'//integer_text(i-1)
      end do
    end if
  end function
  subroutine record_names(stored,p,supplied)
    type(feature_string), allocatable, intent(inout) :: stored(:)
    integer, intent(in) :: p
    type(feature_string), optional, intent(in) :: supplied(:)
    type(feature_string), allocatable :: checked(:)
    if(allocated(stored)) deallocate(stored)
    if(present(supplied)) then
      checked=input_names(p,supplied);stored=checked
    end if
  end subroutine
  logical function reserved_name(name,is_column) result(reserved)
    character(*), intent(in) :: name
    logical, intent(in) :: is_column
    character(25), parameter :: controls(9)=[character(25) :: 'n_jobs','transformer_weights','verbose', &
      'verbose_feature_names_out','transformers','remainder','sparse_threshold','force_int_remainder_cols','transformer_list']
    integer :: i,last
    reserved=.false.;last=4
    if(is_column) last=8
    do i=1,last
      if(same_name(name,trim(controls(i)))) reserved=.true.
    end do
    if(.not.is_column) reserved=reserved.or.same_name(name,'transformer_list')
  end function
  subroutine validate_branches(branches,is_column)
    type(transformer_branch), intent(in) :: branches(:)
    logical, intent(in) :: is_column
    integer :: i,j
    do i=1,size(branches)
      call require(allocated(branches(i)%name),'composition branch name is unallocated')
      call require(index(branches(i)%name,'__')==0,'composition branch name contains __')
      call require(.not.reserved_name(branches(i)%name,is_column),'composition name conflicts with a control')
      do j=1,i-1
        call require(.not.same_name(branches(i)%name,branches(j)%name),'composition branch names must be unique')
      end do
      call validate_branch(branches(i))
    end do
  end subroutine
  subroutine validate_branch(branch)
    type(transformer_branch), intent(in) :: branch
    call require(any(branch%action==['transform       ','drop            ','passthrough     ']), &
      'composition invalid branch action')
    if(branch%action=='transform') then
      call require(allocated(branch%model),'composition branch transformer is unallocated')
      call require(supported(branch%model),'composition unsupported transformer')
    end if
  end subroutine
  recursive logical function supported(model) result(ok)
    class(*), intent(in) :: model
    ok=.true.
    select type(model)
    type is(function_transformer)
    type is(column_transformer)
    type is(feature_union)
    type is(standard_scaler)
    type is(minmax_scaler)
    type is(maxabs_scaler)
    type is(robust_scaler)
    type is(simple_imputer)
    type is(one_hot_encoder)
    type is(ordinal_encoder)
    type is(knn_imputer)
    type is(missing_indicator)
    type is(power_transformer)
    type is(quantile_transformer)
    type is(kbins_discretizer)
    type is(spline_transformer)
    type is(pca)
    type is(truncated_svd)
    type is(variance_threshold)
    class is(dense_transformer)
      ok=model%can_transform()
    class default
      ok=.false.
    end select
  end function
  subroutine clone_model(prototype,model)
    class(*), intent(in) :: prototype
    class(*), allocatable, intent(out) :: model
    type(function_transformer) :: fresh
    select type(prototype)
    type is(function_transformer)
      fresh%func=>prototype%func;fresh%inverse_func=>prototype%inverse_func;fresh%names_func=>prototype%names_func
      fresh%validate=prototype%validate;fresh%check_inverse=prototype%check_inverse
      fresh%feature_names_out=prototype%feature_names_out
      allocate(model,source=fresh)
    class default
      allocate(model,source=prototype)
    end select
  end subroutine
  recursive subroutine fit_model(model,x,names)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: names(:)
    select type(model)
    type is(function_transformer)
      call model%fit(x,names)
    type is(column_transformer)
      call model%fit(x,names)
    type is(feature_union)
      call model%fit(x,names)
    type is(standard_scaler)
      call model%fit(x)
    type is(minmax_scaler)
      call model%fit(x)
    type is(maxabs_scaler)
      call model%fit(x)
    type is(robust_scaler)
      call model%fit(x)
    type is(simple_imputer)
      call model%fit(x)
    type is(one_hot_encoder)
      call model%fit(x)
    type is(ordinal_encoder)
      call model%fit(x)
    type is(knn_imputer)
      call model%fit(x)
    type is(missing_indicator)
      call model%fit(x)
    type is(power_transformer)
      call model%fit(x)
    type is(quantile_transformer)
      call model%fit(x)
    type is(kbins_discretizer)
      call model%fit(x)
    type is(spline_transformer)
      call model%fit(x)
    type is(pca)
      call model%fit(x)
    type is(truncated_svd)
      call model%fit(x)
    type is(variance_threshold)
      call model%fit(x)
    class is(dense_transformer)
      call model%fit_data(x,names)
    class default
      call require(.false.,'composition unsupported transformer')
    end select
  end subroutine
  recursive function fit_transform_model(model,x,names) result(z)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: names(:)
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(column_transformer)
      z=model%fit_transform(x,names)
    type is(feature_union)
      z=model%fit_transform(x,names)
    class is(dense_transformer)
      z=model%fit_map_data(x,names)
    class default
      call fit_model(model,x,names);z=transform_model(model,x)
    end select
  end function
  recursive function transform_model(model,x) result(z)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(function_transformer)
      z=model%transform(x)
    type is(column_transformer)
      z=model%transform(x)
    type is(feature_union)
      z=model%transform(x)
    type is(standard_scaler)
      z=model%transform(x)
    type is(minmax_scaler)
      z=model%transform(x)
    type is(maxabs_scaler)
      z=model%transform(x)
    type is(robust_scaler)
      z=model%transform(x)
    type is(simple_imputer)
      z=model%transform(x)
    type is(one_hot_encoder)
      z=model%transform(x)
    type is(ordinal_encoder)
      z=model%transform(x)
    type is(knn_imputer)
      z=model%transform(x)
    type is(power_transformer)
      z=model%transform(x)
    type is(quantile_transformer)
      z=model%transform(x)
    type is(kbins_discretizer)
      z=model%transform(x)
    type is(spline_transformer)
      z=model%transform(x)
    type is(pca)
      z=model%transform(x)
    type is(truncated_svd)
      z=model%transform(x)
    type is(variance_threshold)
      z=model%transform(x)
    type is(missing_indicator)
      z=merge(1._dp,0._dp,model%transform(x))
    class is(dense_transformer)
      z=model%map_data(x)
    class default
      call require(.false.,'composition unsupported transformer')
    end select
  end function
  recursive function model_names(model,input) result(names)
    class(*), intent(in) :: model
    type(feature_string), intent(in) :: input(:)
    type(feature_string), allocatable :: names(:)
    integer :: i,j,k,n
    character(:), allocatable :: prefix
    select type(model)
    type is(function_transformer)
      names=model%get_feature_names_out(input)
    type is(column_transformer)
      names=model%get_feature_names_out(input)
    type is(feature_union)
      names=model%get_feature_names_out(input)
    type is(pca)
      allocate(names(size(model%components_,1)))
      do i=1,size(names)
        names(i)%text='pca'//integer_text(i-1)
      end do
    type is(truncated_svd)
      allocate(names(size(model%components_,1)))
      do i=1,size(names)
        names(i)%text='truncatedsvd'//integer_text(i-1)
      end do
    type is(one_hot_encoder)
      allocate(names(model%n_features_out_));k=0
      do i=1,size(input)
        do j=1,size(model%categories_(i)%values)
          k=k+1;names(k)%text=input(i)%text//'_'//real_text(model%categories_(i)%values(j))
        end do
      end do
    type is(kbins_discretizer)
      if(model%encode=='ordinal') then
        names=input
      else
        allocate(names(model%n_features_out_));k=0
        do i=1,size(input)
          do j=1,model%n_bins_(i)
            k=k+1;names(k)%text=input(i)%text//'_'//real_text(real(j-1,dp))
          end do
        end do
      end if
    type is(spline_transformer)
      allocate(names(model%n_features_out_));k=0
      n=model%n_features_out_/size(input)
      do i=1,size(input)
        do j=1,n
          k=k+1;names(k)%text=input(i)%text//'_sp_'//integer_text(j-1)
        end do
      end do
    type is(variance_threshold)
      names=pack(input,model%support_)
    type is(missing_indicator)
      allocate(names(size(model%features_)))
      do i=1,size(names)
        names(i)%text='missingindicator_'//input(model%features_(i))%text
      end do
    type is(knn_imputer)
      ! sklearn 1.6.1 names omit invalid columns even with keep_empty_features.
      names=pack(input,model%valid_mask_)
      if(model%add_indicator) then
        do i=1,size(model%indicator_%features_)
          prefix='missingindicator_'//input(model%indicator_%features_(i))%text
          names=[names,feature_string(prefix)]
        end do
      end if
    class is(dense_transformer)
      names=model%feature_names(input)
    class default
      call require(supported(model),'composition unsupported transformer')
      names=input
    end select
  end function
  integer function find_name(names,name) result(index_found)
    type(feature_string), intent(in) :: names(:)
    character(*), intent(in) :: name
    integer :: i
    index_found=0
    do i=1,size(names)
      call require(allocated(names(i)%text),'composition feature name is unallocated')
      if(.not.same_name(names(i)%text,name)) cycle
      call require(index_found==0,'composition selected feature name is not unique')
      index_found=i
    end do
    call require(index_found>0,'composition selected feature name is missing')
  end function
  function selected_columns(branch,x,names,has_names) result(columns)
    type(transformer_branch), intent(in) :: branch
    real(dp), intent(in) :: x(:,:)
    type(feature_string), intent(in) :: names(:)
    logical, intent(in) :: has_names
    integer, allocatable :: columns(:)
    integer :: choices,i
    choices=merge(1,0,allocated(branch%columns))+merge(1,0,allocated(branch%mask))+ &
      merge(1,0,allocated(branch%column_names))+merge(1,0,associated(branch%selector))
    call require(choices==1,'ColumnTransformer requires exactly one column selector per branch')
    if(allocated(branch%columns)) then
      columns=branch%columns
    else if(allocated(branch%mask)) then
      call require(size(branch%mask)==size(x,2),'ColumnTransformer mask width mismatch')
      columns=pack([(i,i=1,size(x,2))],branch%mask)
    else if(allocated(branch%column_names)) then
      call require(has_names,'ColumnTransformer named selection requires feature names')
      allocate(columns(size(branch%column_names)))
      do i=1,size(columns)
        call require(allocated(branch%column_names(i)%text),'ColumnTransformer selected name is unallocated')
        columns(i)=find_name(names,branch%column_names(i)%text)
      end do
    else
      call branch%selector(x,names,columns)
      call require(allocated(columns),'ColumnTransformer selector returned unallocated indices')
    end if
    call require(all(columns>=1).and.all(columns<=size(x,2)),'ColumnTransformer column index out of bounds')
  end function
  subroutine clear_selector(branch)
    type(transformer_branch), intent(inout) :: branch
    if(allocated(branch%mask)) deallocate(branch%mask)
    if(allocated(branch%column_names)) deallocate(branch%column_names)
    nullify(branch%selector)
  end subroutine
  subroutine scale_block(block,weight)
    real(dp), intent(inout) :: block(:,:)
    real(dp), intent(in) :: weight
    integer :: i,j
    do j=1,size(block,2)
      do i=1,size(block,1)
        block(i,j)=ieee_product(block(i,j),weight)
      end do
    end do
  end subroutine
  function concatenate(blocks,n) result(z)
    type(matrix_block), intent(in) :: blocks(:)
    integer, intent(in) :: n
    real(dp), allocatable :: z(:,:)
    integer :: i,start,width,total,rows
    total=0;rows=-1
    do i=1,size(blocks)
      if(.not.allocated(blocks(i)%values)) cycle
      total=total+size(blocks(i)%values,2)
      if(rows<0) rows=size(blocks(i)%values,1)
      call require(size(blocks(i)%values,1)==rows,'composition branch sample counts differ')
    end do
    if(rows<0) rows=n
    allocate(z(rows,total));start=1
    do i=1,size(blocks)
      if(.not.allocated(blocks(i)%values)) cycle
      width=size(blocks(i)%values,2);z(:,start:start+width-1)=blocks(i)%values;start=start+width
    end do
  end function
  recursive subroutine column_fit(self,x,feature_names)
    class(column_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: discarded(:,:)
    discarded=self%fit_transform(x,feature_names)
  end subroutine
  recursive function column_fit_transform(self,x,feature_names) result(z)
    class(column_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    type(feature_string), allocatable :: names(:),local_names(:)
    type(matrix_block), allocatable :: blocks(:)
    type(transformer_branch), allocatable :: fitted(:)
    integer, allocatable :: columns(:),rest(:)
    logical :: used(size(x,2))
    integer :: n,i,j,start,width
    type(function_transformer) :: identity
    class(*), allocatable :: cloned
    self%fitted_=.false.
    if(.not.allocated(self%transformers)) allocate(self%transformers(0))
    call validate_branches(self%transformers,.true.)
    call validate_branch(self%remainder)
    call record_names(self%feature_names_in_,size(x,2),feature_names)
    names=input_names(size(x,2),stored=self%feature_names_in_)
    self%n_features_in_=size(x,2);used=.false.;n=size(self%transformers)
    allocate(fitted(n+1),blocks(n+1))
    do i=1,n
      fitted(i)=self%transformers(i)
      columns=selected_columns(self%transformers(i),x,names,present(feature_names))
      fitted(i)%columns=columns;call clear_selector(fitted(i))
      used(columns)=.true.
    end do
    rest=pack([(j,j=1,size(x,2))],.not.used)
    fitted(n+1)=self%remainder;fitted(n+1)%name='remainder';fitted(n+1)%columns=rest
    call clear_selector(fitted(n+1))
    if(size(rest)>0) n=n+1
    start=1
    do i=1,n
      fitted(i)%output_start_=1;fitted(i)%output_stop_=0
      if(fitted(i)%action=='drop'.or.size(fitted(i)%columns)==0) cycle
      columns=fitted(i)%columns;local_names=names(columns)
      if(fitted(i)%action=='passthrough') then
        identity%feature_names_out='one-to-one'
        if(allocated(fitted(i)%model)) deallocate(fitted(i)%model)
        allocate(fitted(i)%model,source=identity);fitted(i)%action='transform'
      else
        call clone_model(fitted(i)%model,cloned)
        call move_alloc(cloned,fitted(i)%model)
      end if
      if(present(feature_names)) then
        blocks(i)%values=fit_transform_model(fitted(i)%model,x(:,columns),local_names)
      else
        blocks(i)%values=fit_transform_model(fitted(i)%model,x(:,columns))
      end if
      call scale_block(blocks(i)%values,fitted(i)%weight)
      width=size(blocks(i)%values,2);fitted(i)%output_start_=start;fitted(i)%output_stop_=start+width-1
      start=start+width
    end do
    self%transformers_=fitted(:n)
    z=concatenate(blocks,size(x,1));self%n_features_out_=size(z,2)
    self%sparse_output_=.false.;self%fitted_=.true.
  end function
  recursive function column_transform(self,x,feature_names) result(z)
    class(column_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    type(matrix_block), allocatable :: blocks(:)
    type(feature_string), allocatable :: query_names(:)
    integer, allocatable :: columns(:)
    integer :: i,j
    real(dp) :: weight
    call require(self%fitted_,'ColumnTransformer is not fitted')
    if(present(feature_names).and.allocated(self%feature_names_in_)) then
      query_names=input_names(size(x,2),feature_names)
    else
      call require(size(x,2)==self%n_features_in_,'ColumnTransformer query feature width mismatch')
    end if
    allocate(blocks(size(self%transformers_)))
    do i=1,size(blocks)
      associate(branch=>self%transformers_(i))
        if(branch%action=='drop'.or.size(branch%columns)==0) cycle
        columns=branch%columns
        if(allocated(query_names)) then
          do j=1,size(columns)
            columns(j)=find_name(query_names,self%feature_names_in_(branch%columns(j))%text)
          end do
        end if
        blocks(i)%values=transform_model(branch%model,x(:,columns))
        weight=self%remainder%weight
        if(i<=size(self%transformers)) weight=self%transformers(i)%weight
        call scale_block(blocks(i)%values,weight)
      end associate
    end do
    z=concatenate(blocks,size(x,1))
  end function
  integer function column_named_index(self,name) result(found)
    class(column_transformer), intent(in) :: self
    character(*), intent(in) :: name
    integer :: i
    call require(self%fitted_,'ColumnTransformer is not fitted');found=0
    do i=1,size(self%transformers_)
      if(same_name(self%transformers_(i)%name,name)) found=i
    end do
  end function
  function column_output_indices(self,name) result(indices)
    class(column_transformer), intent(in) :: self
    character(*), intent(in) :: name
    integer, allocatable :: indices(:)
    integer :: i,j
    i=self%named_transformer_index(name)
    if(i==0) then
      call require(same_name(name,'remainder'),'ColumnTransformer unknown branch name')
      allocate(indices(0))
    else
      indices=[(j,j=self%transformers_(i)%output_start_,self%transformers_(i)%output_stop_)]
    end if
  end function
  function prefixed_names(branch_name,names,verbose) result(output)
    character(*), intent(in) :: branch_name
    type(feature_string), intent(in) :: names(:)
    logical, intent(in) :: verbose
    type(feature_string), allocatable :: output(:)
    integer :: i
    output=names
    do i=1,size(output)
      call require(allocated(output(i)%text),'composition output feature name is unallocated')
      if(verbose) output(i)%text=branch_name//'__'//output(i)%text
    end do
  end function
  subroutine unique_output_names(names,verbose)
    type(feature_string), intent(in) :: names(:)
    logical, intent(in) :: verbose
    integer :: i,j
    if(verbose) return
    do i=1,size(names)
      do j=1,i-1
        call require(.not.same_name(names(i)%text,names(j)%text),'composition output feature names are not unique')
      end do
    end do
  end subroutine
  recursive function column_names(self,input_features) result(output)
    class(column_transformer), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: output(:),names(:),local(:)
    integer :: i,j
    call require(self%fitted_,'ColumnTransformer is not fitted')
    names=input_names(self%n_features_in_,input_features,self%feature_names_in_)
    allocate(output(0))
    do i=1,size(self%transformers_)
      associate(branch=>self%transformers_(i))
        if(branch%action=='drop'.or.size(branch%columns)==0) cycle
        local=model_names(branch%model,names(branch%columns))
        if(associated(self%name_formatter)) then
          do j=1,size(local)
            call require(allocated(local(j)%text),'composition output feature name is unallocated')
            local(j)%text=self%name_formatter(branch%name,local(j)%text)
          end do
        else
          local=prefixed_names(branch%name,local,self%verbose_feature_names_out)
        end if
        output=[output,local]
      end associate
    end do
    call unique_output_names(output,self%verbose_feature_names_out.or.associated(self%name_formatter))
  end function
  recursive subroutine union_fit(self,x,feature_names)
    class(feature_union), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    integer :: i
    type(function_transformer) :: identity
    self%fitted_=.false.
    if(.not.allocated(self%transformer_list)) allocate(self%transformer_list(0))
    call validate_branches(self%transformer_list,.false.)
    call record_names(self%feature_names_in_,size(x,2),feature_names);self%n_features_in_=size(x,2)
    do i=1,size(self%transformer_list)
      associate(branch=>self%transformer_list(i))
        if(branch%action=='drop') cycle
        if(branch%action=='passthrough') then
          identity%feature_names_out='one-to-one'
          if(allocated(branch%model)) deallocate(branch%model)
          allocate(branch%model,source=identity);branch%action='transform'
        end if
        call fit_model(branch%model,x,feature_names)
      end associate
    end do
    self%n_features_out_=-1;self%fitted_=.true.
  end subroutine
  recursive function union_fit_transform(self,x,feature_names) result(z)
    class(feature_union), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    type(matrix_block), allocatable :: blocks(:)
    integer :: i,start,width
    type(function_transformer) :: identity
    self%fitted_=.false.
    if(.not.allocated(self%transformer_list)) allocate(self%transformer_list(0))
    call validate_branches(self%transformer_list,.false.)
    call record_names(self%feature_names_in_,size(x,2),feature_names);self%n_features_in_=size(x,2)
    allocate(blocks(size(self%transformer_list)));start=1
    do i=1,size(blocks)
      associate(branch=>self%transformer_list(i))
        branch%output_start_=1;branch%output_stop_=0
        if(branch%action=='drop') cycle
        if(branch%action=='passthrough') then
          identity%feature_names_out='one-to-one'
          if(allocated(branch%model)) deallocate(branch%model)
          allocate(branch%model,source=identity);branch%action='transform'
        end if
        blocks(i)%values=fit_transform_model(branch%model,x,feature_names)
        call scale_block(blocks(i)%values,branch%weight)
        width=size(blocks(i)%values,2);branch%output_start_=start;branch%output_stop_=start+width-1
        start=start+width
      end associate
    end do
    z=concatenate(blocks,size(x,1));self%n_features_out_=size(z,2);self%fitted_=.true.
  end function
  recursive function union_transform(self,x) result(z)
    class(feature_union), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    type(matrix_block), allocatable :: blocks(:)
    integer :: i
    call require(self%fitted_,'FeatureUnion is not fitted')
    allocate(blocks(size(self%transformer_list)))
    do i=1,size(blocks)
      associate(branch=>self%transformer_list(i))
        if(branch%action=='drop') cycle
        blocks(i)%values=transform_model(branch%model,x)
        call scale_block(blocks(i)%values,branch%weight)
      end associate
    end do
    z=concatenate(blocks,size(x,1))
  end function
  recursive function union_names(self,input_features) result(output)
    class(feature_union), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: output(:),names(:),local(:)
    integer :: i
    call require(self%fitted_,'FeatureUnion is not fitted')
    names=input_names(self%n_features_in_,input_features,self%feature_names_in_);allocate(output(0))
    do i=1,size(self%transformer_list)
      associate(branch=>self%transformer_list(i))
        if(branch%action=='drop') cycle
        local=model_names(branch%model,names)
        local=prefixed_names(branch%name,local,self%verbose_feature_names_out);output=[output,local]
      end associate
    end do
    call unique_output_names(output,self%verbose_feature_names_out)
  end function
end module
