program reference_sparse_coordinate_driver
  use sklearn_coordinate_paths,only:enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  use sklearn_sparse,only:csr_matrix,csr_from_arrays
  use sklearn_common,only:dp,require
  use,intrinsic::iso_fortran_env,only:int64
  implicit none
  type(csr_matrix)::x,before
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp),allocatable::y(:),alphas(:),initial(:),weights(:),offset(:),scale(:),gram(:,:),xy(:),yb(:)
  real(dp),allocatable::ab(:),ib(:),wb(:),ob(:),sb(:),gb(:,:),cb(:),data(:)
  integer,allocatable::ptr(:),indices(:)
  integer::input,output,cases,c,n,p,nnz,lasso,na,hi,hw,ho,hs,hg,hc,positive,i
  character(1024)::file
  character(40)::prefix
  call get_command_argument(1,file);open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file);open(newunit=output,file=trim(file),status='replace')
  read(input,*)cases
  do c=1,cases
    read(input,*)n,p,nnz,lasso,na,hi,hw,ho,hs,hg,hc
    read(input,*)options%l1_ratio,options%eps,options%n_alphas,options%tol,options%max_iter, &
      options%random_state,positive,options%selection,options%precompute
    options%positive=positive==1
    if(allocated(y))deallocate(y,ptr,indices,data)
    if(allocated(alphas))deallocate(alphas)
    if(allocated(initial))deallocate(initial)
    if(allocated(weights))deallocate(weights)
    if(allocated(offset))deallocate(offset)
    if(allocated(scale))deallocate(scale)
    if(allocated(gram))deallocate(gram)
    if(allocated(xy))deallocate(xy)
    allocate(ptr(n+1),indices(nnz),data(nnz),y(n))
    read(input,*)ptr;read(input,*)indices;read(input,*)data;read(input,*)y
    x=csr_from_arrays(n,p,ptr,indices,data);before=x;yb=y
    if(na>=0)then
      allocate(alphas(na));read(input,*)alphas;ab=alphas
    end if
    if(hi==1)then
      allocate(initial(p));read(input,*)initial;ib=initial
    end if
    if(hw==1)then
      allocate(weights(n));read(input,*)weights;wb=weights
    end if
    if(ho==1)then
      allocate(offset(p));read(input,*)offset;ob=offset
    end if
    if(hs>=0)then
      allocate(scale(hs));read(input,*)scale;sb=scale
    end if
    if(hg==1)then
      ! The source sparse branch discards even a non-square/nonfinite Gram.
      allocate(gram(1,1));read(input,*)gram;gb=gram
    end if
    if(hc==1)then
      allocate(xy(1));read(input,*)xy;cb=xy
    end if
    if(lasso==1)then
      fit=lasso_path(x,y,options,alphas,initial,gram,xy,weights,offset,scale)
    else
      fit=enet_path(x,y,options,alphas,initial,gram,xy,weights,offset,scale)
    end if
    call require(x%n_rows==before%n_rows.and.x%n_cols==before%n_cols,'sparse path shape changed')
    call require(all(x%indptr==before%indptr).and.all(x%indices==before%indices),'sparse path structure changed')
    call retained(x%data,before%data);call retained(y,yb)
    if(allocated(alphas))call retained(alphas,ab)
    if(allocated(initial))call retained(initial,ib)
    if(allocated(weights))call retained(weights,wb)
    if(allocated(offset))call retained(offset,ob)
    if(allocated(scale))call retained(scale,sb)
    if(allocated(gram))call retained(reshape(gram,[size(gram)]),reshape(gb,[size(gb)]))
    if(allocated(xy))call retained(xy,cb)
    write(prefix,'(a,i0,a)')'case',c,'_'
    call emit('shape',real(shape(fit%coefficients),dp));call emit('alphas',fit%alphas)
    call emit('coefficients',reshape(fit%coefficients,[size(fit%coefficients)]))
    call emit('gaps',fit%dual_gaps);call emit('iterations',real(fit%n_iter,dp))
    call emit('converged',real(merge(1,0,fit%converged),dp));call emit('tolerance',[fit%dual_tolerance])
    call require(fit%backend=='sparse','wrong sparse path backend')
    call emit('backend',[3._dp]);call emit('preserved',[1._dp])
  end do
contains
  subroutine retained(a,b)
    real(dp),intent(in)::a(:),b(:)
    call require(size(a)==size(b),'sparse path input extent changed')
    call require(all(transfer(a,[0_int64],size(a))==transfer(b,[0_int64],size(b))), &
      'sparse path input bits changed')
  end subroutine
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    write(output,'(a,*(1x,es25.16e3))')trim(prefix)//name,values
  end subroutine
end program
