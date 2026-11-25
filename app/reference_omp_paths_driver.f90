program reference_omp_paths_driver
  use sklearn_omp_paths,only:orthogonal_mp,orthogonal_mp_gram,omp_result
  use sklearn_common,only:dp,require
  implicit none
  type(omp_result)::fit
  real(dp),allocatable::x(:,:),y(:,:),xb(:,:),yb(:,:),norms(:),norms_before(:),norm_scalar,tolerance
  integer,allocatable::count
  integer::input,output,cases,c,n,p,t,kind,vector,hcount,ntarget,hastol,keep,precompute,i,atoms
  real(dp)::target
  character(1024)::file
  character(40)::prefix
  call get_command_argument(1,file);open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file);open(newunit=output,file=trim(file),status='replace')
  read(input,*)cases
  do c=1,cases
    read(input,*)n,p,t,kind,vector,hcount,atoms,hastol,target,keep,precompute,ntarget
    if(allocated(x))deallocate(x,y)
    if(allocated(count))deallocate(count)
    if(allocated(tolerance))deallocate(tolerance)
    if(allocated(norms))deallocate(norms)
    if(allocated(norm_scalar))deallocate(norm_scalar)
    if(hcount==1)then
      allocate(count);count=atoms
    end if
    if(hastol==1)then
      allocate(tolerance);tolerance=target
    end if
    if(ntarget>=0)then
      allocate(norms(ntarget));read(input,*)norms
      norms_before=norms
      if(ntarget==1)then
        allocate(norm_scalar);norm_scalar=norms(1)
      end if
    end if
    allocate(x(n,p),y(n,t))
    do i=1,n
      read(input,*)x(i,:)
    end do
    do i=1,n
      read(input,*)y(i,:)
    end do
    xb=x;yb=y
    if(kind==1)then
      if(vector==1)then
        fit=orthogonal_mp_gram(x,y(:,1),n_nonzero_coefs=count,tol=tolerance, &
          norms_squared=norm_scalar,return_path=keep==1)
      else
        fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=count,tol=tolerance,norms_squared=norms,return_path=keep==1)
      end if
    else
      if(vector==1)then
        if(precompute==2)then
          fit=orthogonal_mp(x,y(:,1),precompute='auto',n_nonzero_coefs=count,tol=tolerance,return_path=keep==1)
        else
          fit=orthogonal_mp(x,y(:,1),precompute=precompute==1,n_nonzero_coefs=count,tol=tolerance,return_path=keep==1)
        end if
      else
        if(precompute==2)then
          fit=orthogonal_mp(x,y,precompute='auto',n_nonzero_coefs=count,tol=tolerance,return_path=keep==1)
        else
          fit=orthogonal_mp(x,y,precompute=precompute==1,n_nonzero_coefs=count,tol=tolerance,return_path=keep==1)
        end if
      end if
    end if
    call require(all(x==xb).and.all(y==yb),'OMP reference inputs changed')
    if(allocated(norms))call require(all(norms==norms_before),'OMP reference target norms changed')
    write(prefix,'(a,i0,a)')'case',c,'_'
    atoms=0;if(allocated(fit%path))atoms=size(fit%path,3)
    call emit('shape',real([p,t,atoms],dp))
    call emit('coef',reshape(fit%coefficients,[p*t]))
    if(allocated(fit%path))then
      call emit('path',reshape(fit%path,[size(fit%path)]))
    else
      call emit('path',[real(dp)::])
    end if
    call emit('iterations',real(fit%n_iter,dp))
    call emit('premature',real(merge(1,0,fit%premature),dp))
    call emit('backend',[real(merge(1,0,fit%gram_used),dp)])
    call emit('preserved',[1._dp])
  end do
contains
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    write(output,'(a,*(1x,es25.16e3))')trim(prefix)//name,values
  end subroutine
end program
